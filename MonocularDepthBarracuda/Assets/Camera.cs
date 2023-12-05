using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Camera : MonoBehaviour
{
    static WebCamTexture backCam;

    void Start()
    {
        if (backCam == null)
            backCam = new WebCamTexture();

        GetComponent<Renderer>().material.mainTexture = backCam;

        this.GetComponent<RawImage>().material = this.GetComponent<SpriteRenderer>().material;

        if (!backCam.isPlaying)
            backCam.Play();

    }
}
