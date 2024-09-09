def get_center_of_bbox(bbox):
    """
    Calculate the center coordinates of a bounding box.

    Args:
        bbox (list or tuple): Bounding box coordinates in the format [x1, y1, x2, y2], 
                              where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Returns:
        tuple: The (x, y) coordinates of the center of the bounding box.
    """
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)


def get_bbox_width(bbox):
    """
    Calculate the width of a bounding box.

    Args:
        bbox (list or tuple): Bounding box coordinates in the format [x1, y1, x2, y2], 
                              where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Returns:
        int: The width of the bounding box.
    """
    return bbox[2]-bbox[0]