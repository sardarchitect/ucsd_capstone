# VIRAT Dataset
> The VIRAT Video Dataset is designed to be realistic, natural and challenging for video surveillance domains in terms of its resolution, background clutter, diversity in scenes, and human activity/event categories than existing action recognition datasets. It has become a benchmark dataset for the computer vision community.
>
> More info can be found [here](https://viratdata.org/)

## Dataset Summary
- Realism and natural scenes
	- Uncontrolled
	- Cluttered backgrounds
	- Minimized actors
	- People performing normal actions in standard contexts
- Diversity:
	- Collected in the US
- Quantity
	- Release 2.0 includes videos recorded from 11 scenes captured by HD cameras
	- >30 per action class examples
- Wide range of resolution and frame rates (Spatial and temporal)
	- 2 - 30 Hz frame rates
	- 10-200 pixels per person
	- HD and downsampled data available
- Ground and Aerial videos available
- Categories:
	- Single-object
	- Two-objects
	- Around 12 categories per scene
- Additional annotated videos in MEVA dataset

## Technical Information
- Scene Directory Format: VIRAT_S_XXYYZZ
- Filename Format: VIRAT_S_XXYYZZ_KK_SSSSSS_TTTTTT.mp4
	- XX: collection group ID 
	- YY: scene ID  
	- ZZ: sequence ID  
	- KK: segment ID (within sequence)  
	- SSSSSS: starting seconds in %06d format. E.g., 1 min 2 sec is 000062.  
	- TTTTTT: ending seconds in %06d format.
- Vehicle categories:
	- Car
	- Bike
	- Vehicle

## Events Description
### Person loading an Object to a Vehicle
- Description: An object moving from a person to a vehicle. The act of 'carrying' should not be included in this event.
- Annotation: 'Person', 'Object' (optional), and 'Vehicle' should be annotated.
- Start: The event begins immediately when the cargo to be loaded is “extended” toward the vehicle (i.e., before one's posture changes from one of 'carrying', to one of 'loading.').
- End: The event ends after the cargo is placed in the vehicle and person-cargo contact is lost. In the event of an occlusion, it ends when the loss of contact is visible.

### Person Unloading an Object from a Vehicle
- Description: An object moving from a vehicle to a person.
- Annotation: 'Person', 'Object' (optional), and 'Vehicle' should be annotated.
- Start: The event begins immediately when the cargo begins to move. If the start of the event is occluded, it begins when the cargo movement is first visible.
- End: The event ends after the cargo is released. If a person, while holding the cargo, begins to walk away from the vehicle, the event ends (at which time the person is 'carrying'). The event also ends if the vehicle drives away while the person is still in contact with the cargo; after the vehicle has been in motion for more than 2 seconds, the person is 'carrying'.

### Person Opening a Vehicle Trunk
- Description: A person opening a trunk. A trunk is defined as a container specifically designed to store nonhuman cargo on a vehicle. A trunk need not have a lid (i.e., the back of a pickup truck is a trunk), and it need not open from above (i.e., the back of a van, which opens via double doors, is also a trunk).
- Annotation: 'Person', and 'Vehicle' should be annotated with bounding boxes for as many frames as possible during the event duration. The bbox annotation of 'Trunk' is optional.
- Start: The event begins when the trunk starts to move.
- End: The event ends after the trunk has stopped moving.

### Person Closing a Vehicle Trunk
- Description: A person closing a trunk.
- Annotation: 'Person', and 'Vehicle' should be annotated with bounding boxes for as many frames as possible during the event duration. The bbox annotation of 'Trunk' is optional.
- Start: The event begins when the trunk starts to move.
- End: The event ends after the trunk has stopped moving.

### Person getting into a Vehicle
- Description: A person getting into, or mounting (e.g., a motorcycle), a vehicle.
- Annotation: 'Person', and 'Vehicle' should be annotated.
- Start: The event begins when the vehicle's door moves, or, if there is no door, 2s before ½ of the person's body is inside the vehicle.
- End: The event ends when the person is in the vehicle. If the vehicle has a door, the event ends after the door is shut. If not, it ends when the person is in the seated position, or has been inside the vehicle for 2 seconds (whichever comes first).

### Person getting out of a Vehicle
- Description: A person getting out of, or dismounting, a vehicle.
- Annotation: 'Person', and 'Vehicle' should be annotated.
- Start: The event begins when the vehicle's door moves. If the vehicle does not have a door, it begins 2 s before ½ of the person's body is outside the vehicle.
- End: The event ends when standing, walking, or running begins.

### Person gesturing
- Description: A person gesturing. Gesturing is defined as a movement, usually of the body or limbs, which expresses or emphasizes an idea, sentiment, or attitude. Examples of gesturing include pointing, waving, and sign language.
- Annotation: 'Person' should be annotated.
- Start: The event begins when the gesture is evident. For example, when waving, the gesture when the waver begins to raise their arm into the “waving position.”
- End: The event ends when the motion ends

### Person digging (Note: not existing in Release 2.0)
- Description: A person digging. Digging may or may not involve the use of a tool (i.e., digging with one's hands is still considered 'digging'; hands are the tool).
- Annotation: 'Person' should be annotated.
- Start: The event begins when the tool makes contact with the ground.
- End: The event ends 5 s after the tool has been removed from the ground, or immediately if the digging tool is dropped.

### Person Carrying an Object
- Description: A person carrying an object. The object may be carried in either hand, with both hands, or on one's back. Object annotation by bboxes are optional and subject to the difficulty.
- Annotation: 'Person', and 'Object' (optional) are annotated.
- Start: The event begins when the person who will carry the object, makes contact with the object. If someone is carrying an object that is initially occluded, the event begins when the object is visible.
- End: The event ends when the person is no longer supporting the object against gravity, and contact with the object is broken. In the event of an occlusion, it ends when the loss of contact is visible.

### Person running
- Description: A person running for more than 2s.
- Annotation: 'Person' should be annotated.
- Start: When a person is visibly running.
- End: The event will end 2 s after the person is no longer running. If transitioning to Standing, Walking or Sitting the event will end after after Standing, Walking or Sitting.

### Person entering a facility
- Description: A person entering a facility
- Annotation: 'Person' should be annotated.
- Start: The event begins 2 s before that person crosses the facility‘s threshold.
- End: The event ends after the person has completely disappeared from view.

### Person exiting a facility
- Description: A person exiting a facility
- Annotation: 'Person' should be annotated.
- Start: The event begins as soon as the person is visible.
- End: The event ends 2 seconds after the person is completely out of the facility

## Citation
A Large-scale Benchmark Dataset for Event Recognition in Surveillance Video" by Sangmin Oh, Anthony Hoogs, Amitha Perera, Naresh Cuntoor, Chia-Chih Chen, Jong Taek Lee, Saurajit Mukherjee, J.K. Aggarwal, Hyungtae Lee, Larry Davis, Eran Swears, Xiaoyang Wang, Qiang Ji, Kishore Reddy, Mubarak Shah, Carl Vondrick, Hamed Pirsiavash, Deva Ramanan, Jenny Yuen, Antonio Torralba, Bi Song, Anesco Fong, Amit Roy-Chowdhury, and Mita Desai, in Proceedings of IEEE Comptuer Vision and Pattern Recognition (CVPR), 2011.
