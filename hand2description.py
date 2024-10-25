import math

from geom_helper import segment_to_segment_dist, coor_sin


def hand2det(hand_det):
    default_finger_state = {
        "in_fist": False,
        "direction": "right",  # up, upleft, left, downleft, down, downright, right, upright
        "direction_angle": 0
    }
    res = {
        'fingers': {
            'thumb': default_finger_state,
            'index': default_finger_state,
            'middle': default_finger_state,
            'ring': default_finger_state,
            'pinky': default_finger_state,
        },
        'finger_groups': []
    }
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

    palm = [
        hand_det[0],
        hand_det[5],
        hand_det[9],
        hand_det[13],
        hand_det[17],
    ]

    in_fist_delta = 0.05
    in_fist_delta_squared = in_fist_delta * in_fist_delta

    finger_near_delta = 0.04

    def finger2det(finger_start, finger_end):
        res = {
            'in_fist': False,
            'direction': 'right',  # up, upleft, left, downleft, down, downright, right, upright
            'direction_angle': 0
        }
        angle = math.atan2(finger_start.y - finger_end.y, finger_start.x - finger_end.x)
        angle_mapping = {
            'right': (-math.pi / 8, math.pi / 8),
            'upright': (math.pi / 8, math.pi / 8 * 3),
            'up': (math.pi / 8 * 3, math.pi / 8 * 5),
            'upleft': (math.pi / 8 * 5, math.pi / 8 * 7),
            'downleft': (-math.pi / 8 * 7, -math.pi / 8 * 5),
            'down': (-math.pi / 8 * 5, -math.pi / 8 * 3),
            'downright': (-math.pi / 8 * 3, -math.pi / 8),
        }
        if angle >= math.pi / 8 * 7 or angle <= -math.pi / 8 * 7:
            res['direction'] = 'left'
        else:
            for tag in angle_mapping:
                if angle_mapping[tag][0] < angle < angle_mapping[tag][1]:
                    res['direction'] = tag
                    break

        for point in palm:
            x_diff = point.x - finger_end.x
            y_diff = point.y - finger_end.y
            z_diff = point.z - finger_end.z
            dist_squared = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
            if in_fist_delta_squared > dist_squared:
                res["in_fist"] = True
                break

        def finger_inside_point(p1, p2, p3, p4, finger_end):
            res = False
            ax = p1.x - finger_end.x
            ay = p1.y - finger_end.y
            az = p1.z - finger_end.z

            bx = p2.x - finger_end.x
            by = p2.y - finger_end.y
            bz = p2.z - finger_end.z

            cx = 2 * p3.x - p4.x - finger_end.x
            cy = 2 * p3.y - p4.y - finger_end.y
            cz = 2 * p3.z - p4.z - finger_end.z

            absin_pos = (ax * by - ay * bx)
            bcsin_pos = (bx * cy - by * cx)
            casin_pos = (cx * ay - cy * ax)

            res = res or (absin_pos > 0 and bcsin_pos > 0 and casin_pos > 0)
            res = res or (absin_pos < 0 and bcsin_pos < 0 and casin_pos < 0)
            return res

        res["in_fist"] = res["in_fist"] or finger_inside_point(hand_det[0], hand_det[5], hand_det[17], hand_det[17],
                                                               finger_end)
        res["in_fist"] = res["in_fist"] or finger_inside_point(hand_det[0], hand_det[5], hand_det[2], hand_det[2],
                                                               finger_end)
        res["in_fist"] = res["in_fist"] or finger_inside_point(hand_det[0], hand_det[17], hand_det[17], hand_det[9],
                                                               finger_end)

        return res

    res["fingers"]["thumb"] = finger2det(thumb_start, thumb_end)
    res["fingers"]["index"] = finger2det(index_start, index_end)
    res["fingers"]["middle"] = finger2det(middle_start, middle_end)
    res["fingers"]["ring"] = finger2det(ring_start, ring_end)
    res["fingers"]["pinky"] = finger2det(pinky_start, pinky_end)

    fingers_defining_points = {}
    if not res["fingers"]["index"]["in_fist"]:
        fingers_defining_points["index"] = (hand_det[6], hand_det[7])
    if not res["fingers"]["middle"]["in_fist"]:
        fingers_defining_points["middle"] = (hand_det[10], hand_det[11])
    if not res["fingers"]["ring"]["in_fist"]:
        fingers_defining_points["ring"] = (hand_det[14], hand_det[15])
    if not res["fingers"]["pinky"]["in_fist"]:
        fingers_defining_points["pinky"] = (hand_det[19], hand_det[20])

    groups = []
    fing_conn = {}
    fdp = list(fingers_defining_points.keys())
    for _f1 in range(len(fingers_defining_points)):
        for _f2 in range(_f1):
            f1 = fingers_defining_points[fdp[_f1]]
            f2 = fingers_defining_points[fdp[_f2]]
            dist = segment_to_segment_dist(f1[0], f1[1], f2[0], f2[1])
            if dist < finger_near_delta:
                fing_conn[fdp[_f1]] = fing_conn.get(fdp[_f1], set()).union({fdp[_f2]})
                fing_conn[fdp[_f2]] = fing_conn.get(fdp[_f2], set()).union({fdp[_f1]})
    grouped = set()
    for i in fdp:
        if i not in grouped:
            cur_group = {i}
            updated = True
            while updated:
                updated = False
                for elem in cur_group:
                    if cur_group != cur_group.union(fing_conn.get(elem, {elem})):
                        cur_group = cur_group.union(fing_conn.get(elem, {elem}))
                        updated = True
                        break
            grouped = grouped.union(cur_group)
            groups.append(cur_group)
    res['finger_groups'] = groups

    return res


def hand2detection(hand_detection_result):
    res = {
        'right': None,
        'left': None
    }

    for hand_det, hand_side in zip(hand_detection_result.hand_landmarks, hand_detection_result.handedness):
        if hand_side[0].category_name == 'Right':
            res['right'] = hand2det(hand_det)
        if hand_side[0].category_name == 'Left':
            res['left'] = hand2det(hand_det)

    return res


def detection2description(hand_detection):
    if hand_detection is None:
        return "Input is None"
    res = ""
    for finger in hand_detection['fingers']:
        finger_dict = {'thumb': 'Thumb', 'index': 'Index finger', 'middle': 'Middle finger', 'ring': 'Ring finger', 'pinky': 'Pinky finger'}
        direction_dict = {
            'right': 'right',
            'upright': 'up and right',
            'up': 'up',
            'upleft': 'up and left',
            'downleft': 'down and left',
            'down': 'down',
            'downright': 'down and right',
            'left': 'left'
        }
        if finger == 'thumb':
            direction_dict['upright'] = 'right'
            direction_dict['upleft'] = 'left'
            direction_dict['downleft'] = 'left'
            direction_dict['downright'] = 'right'
        finger_data = hand_detection['fingers'][finger]
        finger_desc = finger_dict[finger] + ' is' + ('' if finger_data['in_fist'] else ' not') + ' in fist'
        if not finger_data['in_fist']:
            finger_desc = finger_desc + ' and directed to ' + direction_dict[finger_data['direction']] + ' with angle ' + str(finger_data['direction_angle'])
        res = res + finger_desc + '\n'
    for group in hand_detection['finger_groups']:
        res = res + ("Fingers " if len(group) > 1 else "Finger ")
        for elem in group:
            res = res + elem + ", "
        res = res[:-2] + (" are close to each other\n" if len(group) > 1 else " is alone\n")
    #print(res[:-1])
    return res


def diff2description(diff_buffer):
    res = ""
    x_sum = 0
    y_sum = 0
    coor_buffer = []
    max_sin = 0
    for elem in diff_buffer:
        x_avg = 0
        y_avg = 0
        for component in elem:
            x_avg = x_avg + component.x
            y_avg = y_avg + component.y
        x_avg = x_avg / len(elem)
        y_avg = y_avg / len(elem)
        x_sum = x_sum + x_avg
        y_sum = y_sum + y_avg
        coor_buffer.append((x_sum, y_sum))
    for elem in coor_buffer:
        traj_sin = abs(coor_sin(elem, coor_buffer[-1]))
        max_sin = max(max_sin, traj_sin)
    if max_sin > 1/2:
        res = res + "Trajectory is curved"
    else:
        res = res + "Trajectory is straight"
    if abs(x_sum) > abs(y_sum):
        if x_sum > 0:
            res = res + "and directed to the right"
        else:
            res = res + "and directed to the left"
    else:
        if y_sum > 0:
            res = res + "and directed to the up"
        else:
            res = res + "and directed to the down"
    res = res + "\n"
    return res


def actions2description(actions, diff_buffer):
    res = ""
    last_action = None
    cnt = 1
    for action in actions:
        hand_detection = hand2det(action['landmarks'])
        hand_description = detection2description(hand_detection)
        if action['static']:
            res = res + f"Static state number {cnt}:\n" + hand_description
            cnt = cnt + 1
        else:
            res = res + "Trajectory from one gesture to another:\n" + diff2description(diff_buffer[last_action['state_index']:action['state_index']])
        last_action = action
    return res

