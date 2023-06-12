# TODO
- `setuptools`

## Quickstart/Setup
- all the reqs, dependencies, 

## Usage

## Training

## Fine-tuning

## Examples


# UCSD Capstone Project Proposal
## Problem statement
As urban areas become increasingly populated, architects and planners face the challenge of designing spaces that can accommodate high volumes of foot traffic. Furthermore, urban spaces are used in ways usually not intended by the designers. For instance, pedestrians may choose a shortcut through the grass rather than following the paved pathway provided by the designer if it is more efficient. 

By optimizing our environment to behave in ways that are more attuned to pedestrian needs, we can create better public spaces. Moreover, this optimization can allow for more nature to flourish in our cities. For example, by finding patterns in the paths pedestrians take, designers can reduce the need for large slabs of concrete that destroy biodiversity which would have thrived in its place.

The redesign of public spaces with pedestrian needs in mind offers an opportunity to create sustainable, attractive urban areas that promote both biodiversity and human well-being. 

While there are existing products that provide analysis of pedestrian foot traffic across a neighborhood, they mostly use mobile data to get an aggregation of street usage. There are other products that help track customer desnity, but they are optimized for controlled, indoor enviornmnets like grocery stores. Having a computer vision model that analyses pedestrian traffic on specific sections of an outdoor environment, for example a plaza, will provide designers a granular analysis of how the space is currently being used.


## Objectives
To aid in this effort, I propose the development of a computer vision model that can analyze outdoor pedestrian foot traffic patterns, including the number of pedestrians, their direction of movement, the speed at which they are moving, and behaviors they are most likely performing. This information can be used by architects and urban planners to design more pedestrian-friendly spaces, with better flow, accessibility, sustainability, and safety. 

## Methodology
The computer vision model will be developed using a supervised deep learning approach, leveraging state-of-the-art algorithms and techniques in object detection and tracking. The model will be trained on a large dataset of outdoor pedestrian areas, including sidewalks, plazas, and parks, with varying levels of foot traffic and environmental conditions such as weather and time of day. The dataset will be annotated with ground-truth foot traffic patterns to facilitate the supervised approach. If feasible, the annotation will also include pedestrian behaviors. The model will be able to handle a variety of outdoor lighting conditions, such as shadows and reflections, and be able to detect and track pedestrians on different types of surfaces, such as concrete, grass, and gravel. 

The model will also be able to analyze foot traffic patterns over time, providing insights on peak foot traffic times and other patterns that can be used to inform design decisions. The output of the computer vision model will be in a format that can be easily integrated into architectural and urban planning software, allowing architects and planners to use the foot traffic data as input to their designs.

## Data collection and preprocessing
The computer vision model would require a large dataset of outdoor pedestrian areas, including sidewalks, plazas, and parks, with varying levels of foot traffic and environmental conditions such as weather and time of day, along with annotations that establish ground-truth foot traffic patterns (directly or indirectly), and if feasible, pedestrian behavior to facilitate better predictions.

A preliminary review has shown that the VIRAT dataset meets the above mentioned requirements. There is a chance that this dataset might need further cleaning and preprocessing such as resizing, cropping, and normalizing the images to ensure that they are suitable for input into the computer vision model. After this, the model shall be able to detect pedestrian locations over time, along with their behaviors. To take a step further, feature the dataset can be extended to include a perspective grid on top of the scene to help the model seek areas of interest. 

In addition to the above mentioned dataset, there are several others I shall consider. Some of those are:
1. CityPersons: A large-scale pedestrian detection and tracking dataset that contains over 200,000 annotated pedestrian instances in various urban scenes, including crowded pedestrian areas, streets, and shopping malls.
2. MOT (Multiple Object Tracking) Challenge Datasets: A series of benchmark datasets for evaluating multiple object tracking algorithms. The datasets contain video footage of various pedestrian scenarios, including crowded scenes, indoor environments, and outdoor areas.
3. DukeMTMC: A multi-camera pedestrian tracking dataset that contains video footage from 8 cameras in a busy outdoor area on the Duke University campus. The dataset includes over 2.5 million pedestrian trajectories and is suitable for evaluating multi-camera tracking algorithms.
4. ETHZ Pedestrian Dataset: A dataset of pedestrian trajectories captured from a bird's-eye view in an outdoor urban area. The dataset contains over 2,000 pedestrian trajectories and is suitable for evaluating pedestrian tracking algorithms in outdoor environments.

## Model Selection and Training
I will test multiple state-of-the-art computer vision models that are capable of detecting and tracking pedestrians in an outdoor environment. Some popular models I will experiment with include Faster R-CNN, YOLOv8, and Mask R-CNN. The dataset shall be divided into train, test, and evaluation, the exact division size shall be determined depending on the dataset used, and compute available. I will also create a training pipeline that shoud allow me to get training metrics during training, and fine-tune the model on the dataset by adjusting the model's hyperparameters and optimizing its performance.:

One model shall be trained for pedestrians detection, another for tracking, and the third for predicting their behaviors. At the same time, a logger will be used to aggregate pedestrian path densities and behaviors that shall be used as the output to visualize (for example, on a heatmap) how pedestrians seem to use this space.  

## Evaluation metrics
Evaluating the pedestrian detection, tracking, and behaviors will be through using straight-forward supervised learning techniques like Average Precision (AP) and mean Average Precision (mAP), which use Intersection over Union (IoU) to measure precision/recall.   

## Infrastructure Requirements
All experimentation shall be performed on my local machines, but I will use Paperspace to perform the final training.

## Conclusion
In conclusion, the proposed computer vision model aims to solve the challenge of designing pedestrian-friendly urban areas that promote biodiversity and human well-being. By analyzing foot traffic patterns, architects and planners can optimize the environment to better suit pedestrian needs, which could lead to more sustainable and attractive urban spaces.

The development of this model will involve leveraging state-of-the-art algorithms and techniques in object detection and tracking, using a supervised deep learning approach. The model will be trained on a large dataset of outdoor pedestrian areas, including sidewalks, plazas, and parks, and will be able to handle a variety of environmental conditions such as weather and time of day.

Data collection and preprocessing will be done using publicly available datasets such as VIRAT, CityPersons, MOT Challenge, DukeMTMC, and ETHZ Pedestrian. Multiple state-of-the-art models will be tested and evaluated using metrics such as average precision and accuracy.

Overall, the proposed computer vision model has the potential to revolutionize the way we design and plan urban areas, by providing granular insights on foot traffic patterns that can be used to optimize the environment for pedestrian needs, while promoting biodiversity and human well-being.