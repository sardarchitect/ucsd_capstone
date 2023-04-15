events_columns = {
    0: "event_id",        
    1: "event_type",      
    2: "duration", 
    3: "start_frame",     
    4: "end_frame",       
    5: "current_frame",   
    6: "bbox_lefttop_x", 
    7: "bbox_lefttop_y", 
    8: "bbox_width",      
    9: "bbox_height",    
}

objects_columns = {
    0: "object_id",        
    1: "object_duration", 
    2: "current_frame",
    3: "bbox_lefttop_x", 
    4: "bbox_lefttop_y", 
    5: "bbox_width",      
    6: "bbox_height",
    7: "object_type"
}

mapping_columns = {
0: "event_id",
1: "event_type",
2: "event_duration",
3: "start_frame",
4: "end_frame",
5: "num_obj" ,
}

events_type_id_dict = {
    1: "Person loading an Object to a Vehicle",
    2: "Person Unloading an Object from a Car/Vehicle",
    3: "Person Opening a Vehicle/Car Trunk",
    4: "Person Closing a Vehicle/Car Trunk",
    5: "Person getting into a Vehicle",
    6: "Person getting out of a Vehicle",
    7: "Person gesturing",
    8: "Person digging",
    9: "Person carrying an object",
    10: "Person running",
    11: "Person entering a facility",
    12: "Person exiting a facility",
}

objects_type_id_dict = {
    1: "person",
    2: "car",
    3: "vehicles",
    4: "object",
    5: "bike, bicycles"
}