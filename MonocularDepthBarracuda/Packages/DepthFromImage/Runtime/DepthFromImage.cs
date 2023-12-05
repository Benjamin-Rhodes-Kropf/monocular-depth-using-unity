using Unity.Barracuda;
using UnityEngine;
using UnityEngine.Events;
using static Unity.Barracuda.BarracudaTextureUtils;

namespace UnchartedLimbo.NN.Depth
{
    public class DepthFromImage : MonoBehaviour
    {
        [Header("Object References")]
        public NNModel neuralNetworkModel;
        public Texture       inputTexture;
        private WebCamTexture _webCamTexture;

        [Header("Parameters")]
        public bool calculateDepthExtents;
        public DepthMesher depthMesher; // Reference to your DepthMesher instance
        private readonly int desiredWidth = 256;  // Model's desired width
        private readonly int desiredHeight = 256; // Model's desired height

        [Header("Events")]
        public UnityEvent<RenderTexture> OnColorReady;
        public UnityEvent<RenderTexture> OnDepthSolved;
        public UnityEvent<float>         OnImageResized;
        public UnityEvent<Vector2>       OnDepthExtentsCalculated;
        private RenderTexture _resizedTexture;
        
        public Texture InputTexture
        {
            get => inputTexture;
            set => inputTexture = value;
        }
        
        public Texture ColorTexture
        {
            get { return _webCamTexture; } // Assuming _webCamTexture is your camera feed
        }

        private Model         _model;
        private IWorker       _engine;
        private RenderTexture _input, _output;
        private int           _width, _height;

        private void Start()
        {
            // Initialize and start the webcam with desired resolution
            _webCamTexture = new WebCamTexture(desiredWidth, desiredHeight);
            _webCamTexture.Play();

            InitializeNetwork();
            AllocateObjects();
        }

        private void Update()
        {
            if (_webCamTexture == null)
            {
                Debug.Log("WebCamTexture is Null");
                return;
            }

            // Ensure _resizedTexture is allocated with the desired dimensions
            if (_resizedTexture == null || _resizedTexture.width != desiredWidth || _resizedTexture.height != desiredHeight)
            {
                if (_resizedTexture != null)
                {
                    _resizedTexture.Release();
                }
                _resizedTexture = new RenderTexture(desiredWidth, desiredHeight, 0, RenderTextureFormat.ARGB32);
                _resizedTexture.Create();
            }

            // Resize the webcam texture
            Graphics.Blit(_webCamTexture, _resizedTexture);

            // Set the resized texture as input
            inputTexture = _resizedTexture;

            // Ensure _input is allocated with the correct dimensions
            if (_input == null || _input.width != desiredWidth || _input.height != desiredHeight)
            {
                if (_input != null)
                {
                    _input.Release();
                }
                _input = new RenderTexture(desiredWidth, desiredHeight, 0, RenderTextureFormat.ARGB32);
                _input.Create();
            }

            // Fast resize
            Graphics.Blit(inputTexture, _input);

            OnColorReady.Invoke(_input);

            if (neuralNetworkModel == null)
            {
                Debug.Log("NN model is Null");
                return;
            }
            
            RunModel(_input);

            OnImageResized.Invoke(inputTexture.height / (float) inputTexture.width);
            OnDepthSolved.Invoke(_output);
            
            if(depthMesher != null)
            {
                depthMesher.OnColorReceived(ColorTexture);
            }
        }

        private void OnDestroy() => DeallocateObjects();

        /// <summary>
        /// Loads the <see cref="NNModel"/> asset in memory and creates a Barracuda <see cref="IWorker"/>
        /// </summary>
        private void InitializeNetwork()
        {
            if (neuralNetworkModel == null)
                return;

            // Load the model to memory
            _model = ModelLoader.Load(neuralNetworkModel);

            // Create a worker
            _engine = WorkerFactory.CreateWorker(_model, WorkerFactory.Device.GPU);

            // Get Tensor dimensionality ( texture width/height )
            // In Barracuda 1.0.4 the width and height are in channels 1 & 2.
            // In later versions in channels 5 & 6
            #if _CHANNEL_SWAP
                _width  = _model.inputs[0].shape[5];
                _height = _model.inputs[0].shape[6];
            #else
                _width  = _model.inputs[0].shape[1];
                _height = _model.inputs[0].shape[2];
            #endif
        }

        /// <summary>
        /// Allocates the necessary <see cref="RenderTexture"/> objects.
        /// </summary>
        private void AllocateObjects()
        {
            // Check for accidental memory leaks
            if (_input != null) _input.Release();
            if (_output != null) _output.Release();

            Debug.Log("Render Texture Size: (" + desiredWidth + ", " + desiredHeight + ")");

            // Declare texture resources
            _input = new RenderTexture(desiredWidth, desiredHeight, 0, RenderTextureFormat.ARGB32);
            _output = new RenderTexture(desiredWidth, desiredHeight, 0, RenderTextureFormat.RFloat); // Assuming RFloat is the desired format

            // Initialize memory
            _input.Create();
            _output.Create();
        }

        /// <summary>
        /// Releases all unmanaged objects
        /// </summary>
        private void DeallocateObjects()
        {
            _engine?.Dispose();
            _engine = null;

            if (_input != null) _input.Release();
            _input = null;

            if (_output != null) _output.Release();
            _output = null;

        }

        /// <summary>
        /// Performs Inference on the Neural Network Model
        /// </summary>
        /// <param name="source"></param>
        private void RunModel(Texture source)
        {
            using (var tensor = new Tensor(source, 3))
            {
                _engine.Execute(tensor);
            }
            
            // In Barracuda 1.0.4 the output of MiDaS can be passed  directly to a texture as it is shaped correctly.
            // In later versions we have to reshape the tensor. Don't ask why...
            #if _CHANNEL_SWAP
                var to = _engine.PeekOutput().Reshape(new TensorShape(1, _width, _height, 1));
            #else
                 var to = _engine.PeekOutput();
            #endif
         
              TensorToRenderTexture(to, _output, fromChannel:0);


              if (calculateDepthExtents)
              {
                  var data     = to.data.SharedAccess(out var o);
                  var minDepth = data.Min();
                  var maxDepth = data.Max();  
                  OnDepthExtentsCalculated.Invoke(new Vector2(minDepth,maxDepth));
              }
            

            to?.Dispose();
        }


    }
}
