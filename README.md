# NanoNet: Real-Time Polyp Segmentation in Video Capsule Endoscopy and Colonoscopy

The architecture of the NanoNet follows an encoder-decoder approach as shown in Figure 1. As depicted in  Figure  below, the network architecture uses a  pre-trained model as an encoder, followed by the three decoder blocks. Using pre-trained Ima-geNet  models for transfer learning has become the best choice for many CNN architecture. It helps the model converge much faster and achieve high-performance compared to the non-pre-trained model. The proposed architecture uses a MobileNetV2 model pre-trained on the ImageNet dataset as the encoder.  The decoder is built using a  modified version of the residual block, which was initially introduced by He et al. The encoder is used to capture the required contextual information from the input,  whereas the decoder is used to generate the final output by using the contextual information extracted by the encoder.

[NanoNet: Real-Time Polyp Segmentation in Endoscopy](nanonet.pdf).


## Proposed architecture

<img src="figures/nanonet.png">


## Results

### Quantitative Results
<img src="figures/quantative.png">

### Qualitative Results
<img src="figures/qualitative.png">

## Citation
Please cite our paper if you find the work useful: 
<pre>
@inproceedings{jha2021nanonet,
  title={Nanonet: Real-time polyp segmentation in video capsule endoscopy and colonoscopy},
  author={Jha, Debesh and Tomar, Nikhil Kumar and Ali, Sharib and Riegler, Michael A and Johansen, H{\aa}vard D and Johansen, Dag and de Lange, Thomas and Halvorsen, P{\aa}l},
  booktitle={Proceedings of the IEEE 34th International Symposium on Computer-Based Medical Systems (CBMS)},
  pages={37--43},
  year={2021},
  organization={IEEE}
}
}
</pre>

## Contact
Please contact debesh@simula.no, nikhilroxtomar@gmail.com, sharib.ali@eng.ox.ac.uk, michael@simula.no for any further questions.
