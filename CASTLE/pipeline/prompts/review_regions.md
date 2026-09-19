In addition to the main annotation JSON, return a top-level review_regions array.
Use at most MAX_REVIEW_REGIONS regions across the whole clip. Each item:
{"frame_index":0,"box_2d":[0,0,1000,1000],"label":"a concise target","reason":"the detail that could change this annotation"}

frame_index is a supplied frame index. box_2d is normalized [top,left,bottom,right] in 0..1000 coordinates on the ORIGINAL VIDEO CONTENT, excluding the added timestamp footer. Select only uncertain, consequential readable text, hand-object contact, appliance controls, or object-state detail worth inspecting at native resolution. Do not request face identity crops. An empty array is correct if a second look is unlikely to help. A crop cannot recover unrecorded motion between one-second samples; flag that limitation instead of requesting more pixels.
