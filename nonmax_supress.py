
def nonmax_supress(bboxes, scores):

    sorted_idx = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    my_list = [];
    is_valid_bbox = [1] * len(bboxes)

    for i in range(0,len(sorted_idx)):
        curr_bbox = bboxes[sorted_idx[i]]
        center = [curr_bbox[0]+((curr_bbox[2] - curr_bbox[0])/2) , curr_bbox[1]+((curr_bbox[3] - curr_bbox[1])/2)]
        for j in range(i+1,len(sorted_idx)):
            if is_valid_bbox[sorted_idx[j]] != 0:
                my_bbox = bboxes[sorted_idx[j]]
                if center[0] >= my_bbox[0] and center[0] <= my_bbox[2] and center[1] >= my_bbox[1] and center[1] <= my_bbox[3]:
                    is_valid_bbox[sorted_idx[j]] = 0;
               
    max_bboxes = []
    for i in range(0, len(is_valid_bbox)):
        if is_valid_bbox[i] == 1:
            max_bboxes.append(bboxes[i])
            
    return max_bboxes