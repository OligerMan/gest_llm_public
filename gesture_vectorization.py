from mediapipe.framework.formats import landmark_pb2

from geom_helper import dist_p2p


def hand2vec(hand_det):

    thumb_start = hand_det[2]
    index_start = hand_det[5]
    middle_start = hand_det[9]
    ring_start = hand_det[13]
    pinky_start = hand_det[17]

    thumb_end = hand_det[4]
    index_end = hand_det[8]
    middle_end = hand_det[12]
    ring_end = hand_det[16]
    pinky_end = hand_det[20]

    hand_center = hand_det[0]
    elevating_const = 2
    hand_center_elevated = landmark_pb2.NormalizedLandmark(x=hand_det[0].x, y=hand_det[0].y + elevating_const, z=hand_det[0].z)

    thumb_base_dist = dist_p2p(hand_center, thumb_start)
    index_base_dist = dist_p2p(hand_center, index_start)
    middle_base_dist = dist_p2p(hand_center, middle_start)
    ring_base_dist = dist_p2p(hand_center, ring_start)
    pinky_base_dist = dist_p2p(hand_center, pinky_start)

    thumb2index_base_dist = dist_p2p(index_start, thumb_start)
    index2middle_base_dist = dist_p2p(middle_start, index_start)
    middle2ring_base_dist = dist_p2p(ring_start, middle_start)
    ring2pinky_base_dist = dist_p2p(pinky_start, ring_start)

    scaling_parameters = [
        thumb_base_dist,
        index_base_dist,
        middle_base_dist,
        ring_base_dist,
        pinky_base_dist,

        thumb2index_base_dist,
        index2middle_base_dist,
        middle2ring_base_dist,
        ring2pinky_base_dist
    ]

    scaling_factor = 0
    for param in scaling_parameters:
        scaling_factor = scaling_factor + param
    scaling_factor = scaling_factor / len(scaling_parameters)

    thumb2index_end_dist = dist_p2p(thumb_end, index_end)
    index2middle_end_dist = dist_p2p(index_end, middle_end)
    middle2ring_end_dist = dist_p2p(middle_end, ring_end)
    ring2pinky_end_dist = dist_p2p(ring_end, pinky_end)

    thumb_end_dist = dist_p2p(thumb_end, thumb_start)
    index_end_dist = dist_p2p(index_end, index_start)
    middle_end_dist = dist_p2p(middle_end, middle_start)
    ring_end_dist = dist_p2p(ring_end, ring_start)
    pinky_end_dist = dist_p2p(pinky_end, pinky_start)

    thumb_turn_dist = dist_p2p(thumb_end, hand_center_elevated)
    index_turn_dist = dist_p2p(index_end, hand_center_elevated)
    middle_turn_dist = dist_p2p(middle_end, hand_center_elevated)
    ring_turn_dist = dist_p2p(ring_end, hand_center_elevated)
    pinky_turn_dist = dist_p2p(pinky_end, hand_center_elevated)

    vector = [
        thumb2index_end_dist / scaling_factor,
        index2middle_end_dist / scaling_factor,
        middle2ring_end_dist / scaling_factor,
        ring2pinky_end_dist / scaling_factor,

        thumb_end_dist / scaling_factor,
        index_end_dist / scaling_factor,
        middle_end_dist / scaling_factor,
        ring_end_dist / scaling_factor,
        pinky_end_dist / scaling_factor,

        thumb_turn_dist / scaling_factor,
        index_turn_dist / scaling_factor,
        middle_turn_dist / scaling_factor,
        ring_turn_dist / scaling_factor,
        pinky_turn_dist / scaling_factor,
    ]
    return vector
