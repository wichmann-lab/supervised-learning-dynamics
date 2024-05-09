# Data and code from "Comparing supervised learning dynamics: Deep neural networks match human data efficiency but show a generalisation lag"
	
 ![This is an image](https://github.com/wichmann-lab/supervised-learning-dynamics/blob/main/plots/color_signature.png)
	
This repository contains data and code from the paper [Comparing supervised learning dynamics: Deep neural networks match human data efficiency but show a generalisation lag](https://openreview.net/pdf?id=yb9LLnUdqU) that appeared at the ICLR 2024 Workshop on [Representational Alignment (Re-Align)](https://representational-alignment.github.io/#cfp). In the presented study we designed a constrained learning environment with aligned learning conditions to provide a side-by-side comparison of supervised representation learning in humans and various classic CNNs and state-of-the-art (SOTA) deep learning models. Our findings indicate that, under matched learning conditions, DNNs demonstrate a level of data efficiency comparable to human learners, challenging some prevailing assumptions in the field. However, comparisons across the entire learning process also reveal representational differences: while DNNs' learning is characterized by a pronounced generalisation lag, humans appear to immediately acquire generalizable representations without a preliminary phase of learning training set-specific information that is only later transferred to novel data.
	
Please feel free to contact me at lukas.s.huber@unibe.ch or open an issue in case there is any question! 
	
This README is structured according to the repo's structure: one section per subdirectory (alphabetically).
	
## analysis
	
Each script in the `analysis/` directory corresponds to an analysis reported in the paper. Corresponding figures in the paper are indicated in the file name. E.g., `analysis/gen_lag_f3b.py` provides the analysis reported in Figure 3b. All plots reported in the paper can be generated with these scripts and are stored in the `plots/` directory.
	
## data 
	
	
The `data/` directory contains the human and the DNN data obtained in the learning task. Here's what the collum names in the `.csv` files stand for: 

- __run__: For DNNs; number of the fine-tuning run. Each model was fine-tuned for 20 individual runs, each time initialized with the pre-trained ImageNet1k weights. For human observers; subject number.
- __epoch__: Epoch number. For each run or subject there are six epochs. 
- __image_name__: Identifier for the presented image in a particular trial
- __ground_truth__: The presented (ground truth) category.
- __prediction__: The response given by the observer. I.e., the category which the observer "thinks" corresponds to the shown image.
	
	
## dnns
	
In the `dnns` directory, you find the code used to fine-tune the DNN models as well as the dataset employed for fine-tuning. The `dnns/main.py` script fine-tunes and evaluates the following models from the [PyTorch Model Zoo](https://pytorch.org/vision/stable/models.html): [ResNet50](https://arxiv.org/pdf/1512.03385), [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), [VGG16](https://arxiv.org/abs/1409.1556), [ConvNeXt](https://arxiv.org/pdf/2201.03545), and [EfficientNet](https://arxiv.org/pdf/2104.00298). Somehow, I couldn't get the fine-tuning to work for the [ViT](https://arxiv.org/abs/2010.11929) model within the PyTorch framework. Therefore, I used the [Hugging Face](https://huggingface.co/google/vit-base-patch16-224) model library to fine-tune  the ViT model. The corresponding script can be found in `dnns/vit.py`. 
	
The `dnns/dataset` directory contains the training dataset as well as six different test datasets employed to train and evaluate both, humans and DNN models. The stimuli were created using the [Digital Embryo Workshop](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3598413/) created by Karin Hauffen, Eugene Bart, Mark Brady, Daniel Kersten, and Jay Hegdé. For more information on the stimulus, see Section 2.1 in the paper. Each image (e.g., `second_gen_v_15_z_-30.png`) name is a concatenation of the following information (separated by '_'):
	
1. & 2. Generation of the object used for this stimulus. All objects used for the learning task are second-generation objects
3. Category of the object used for this stimulus. `p` is category Eulf, `q` is Puns, and `v` is Lauz
3. Object identifier within the category
4. Rotation axis for the rendering
5. Rotation angle
	
## plots
	
The `plots/` directory contains all plots reported in the paper and can be generated using the code from the `analysis/` directory.
	
## citation
	
@article{huber2022developmental,
	title={The developmental trajectory of object recognition robustness: children are like small adults but unlike big deep neural networks},
	author={Huber, Lukas S and Geirhos, Robert and Wichmann, Felix A},
	journal={arXiv preprint arXiv:2205.10144},
	year={2022}
    }
