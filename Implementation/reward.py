# The TARGET_NUMBER_STEPS is a way to weight the progress over other factors.  progress really should
# be the most important feature, and we don't want race-line hunting to overshadow the progress by
# them being orders of magnitude off.  150 steps is a fair pace on the Toronto track, so below that
# the heading and race-line proximity are dominant factors.  Above 150 progress pace, the car will
# get progress as the dominant reward factor.
TARGET_NUMBER_STEPS = 200
# AWS_RACE_LINE length 69 == 500 steps; CANADA_RACE_LINE length 21 == 150 steps
# LONDON_RACE_LINE, BOWTIE_RACE_LINE, NEW_YORK_RACE_LINE, OVAL_RACE_LINE, CANADA_RACE_LINE, TOKYO_RACE_LINE, AWS_RACE_LINE, REINVENT_2019_RACE_LINE
RACE_LINE_WAYPOINTS = REINVENT_2019_RACE_LINE

MAX_STEERING_ANGLE = 30

# Globals
g_last_progress_value = 0.0
g_start_offset_percent = 0.0
g_race_line_string = LineString(RACE_LINE_WAYPOINTS)
# for getting current race line waypoint distances
g_race_line_dists = [LineString(RACE_LINE_WAYPOINTS).project(Point(p), normalized=True) for p in LineString(RACE_LINE_WAYPOINTS).coords[:-1]] + [1.0]
g_highest_speed = 0.0
g_moving_average_speed = 0.0
g_last_position = (0.0,0.0)

#===============================================================================
#
# REWARD
#
#===============================================================================

def reward_function(params):
    p = progress_factor(params)
    h = race_line_heading_factor(params)
#    d = race_line_distance_factor(params)
    e = edge_factor(params)
    s = steering_factor(params)

    print("progress_factor ", p, " heading_factor", h, " steering_factor ", s)
    reward = p * \
             apply_weight(h, 0.25) * \
             apply_weight(s, 0.25)
    return float(max(reward, 1e-3)) # make sure we never return exactly zero

#===============================================================================
# PROGRESS
#===============================================================================

# Using the race-line coords, calculate progress the same way
# that params['progress'] is calculated (without support for reverse direction)
def current_progress_along_race_line(params):
    global g_start_offset_percent
    global g_race_line_string

    #print("params ", params)
    race_line_position_percent = g_race_line_string.project(Point(params['x'], params['y']), normalized=True)
    #print("race_line_position_percent (absolute): ", race_line_position_percent)

    # reset the "zero" position along the race line string
    # We add params['progress'] here  since
    #    a) it's always non-zero for the standard progress, so the number will just be closer
    #    b) if its near zero, we risk some small chance of it being negative and 
    #       screwing up calculations of rewards
    if params['steps'] == 0:
        g_start_offset_percent = race_line_position_percent - (params['progress'] / 100.0)
        #print("Resetting zero-position to ", g_start_offset_percent )

    race_line_progress_percent = race_line_position_percent - g_start_offset_percent
    if race_line_progress_percent < 0.0:
        race_line_progress_percent = race_line_progress_percent + 1.0
    #print("race_line_progress_percent (relative to start): ", race_line_progress_percent)
    race_line_progress_hundreds = 100 * race_line_progress_percent
    return race_line_progress_hundreds
    
def progress_factor(params):
    # Progress range:  0..100
    # Step is roughly a 1/15s timeslice so can account for time-factor
    # Expected real value: [0,~1.0]
    global g_last_progress_value
    
    # Simple reward for outlier case of first step in the episode
    if params['steps'] == 0:
        g_last_progress_value = 0.0

    # Find the source of progress
    #progress_hundreds = params['progress'] # Use this for centerline progress
    progress_hundreds = current_progress_along_race_line(params) # Use this for race-line progress
    #print("progress_hundreds ", progress_hundreds)
    
    # Do rewards calculation based on progress delta
    delta = progress_hundreds - g_last_progress_value
    g_last_progress_value = progress_hundreds
    
    progress_target_per_step = 100.0 / TARGET_NUMBER_STEPS # use 1 instead of 100 to match progress_since_last magnitude
    progress_factor = delta / progress_target_per_step
    return progress_factor


#===============================================================================
# STEERING
#===============================================================================

def steering_factor(params):
    global MAX_STEERING_ANGLE
    steering_severity = abs(params['steering_angle']) / MAX_STEERING_ANGLE
    return max(min(1.0 - steering_severity, 1.0), 0.0)

   
#===============================================================================
# HEADING
#===============================================================================

def race_line_heading_factor(params):
    global RACE_LINE_WAYPOINTS
    global g_race_line_string
    global g_race_line_dists
    
    # Find the nearest waypoints. Environment does this for us w.r.t. center line,
    # so we repeat it here for race-line
    current_position = Point(params['x'], params['y'])
    current_ndist = g_race_line_string.project(current_position, normalized=True)

    next_index = bisect.bisect_right(g_race_line_dists, current_ndist)
    prev_index = next_index - 1
    if next_index == len(g_race_line_dists):
        next_index = 0
        
    # Target heading in euler reference coordinates
    target_heading = angle_of_vector(RACE_LINE_WAYPOINTS[prev_index], RACE_LINE_WAYPOINTS[next_index])

    heading_delta = abs(target_heading - params['heading'])
    if heading_delta > 180: 
        heading_delta = 360 - heading_delta

    # Gradient factor from 1.0 to 0.0, with
    #  1.0 :  delta is zero
    #  0.0 :  delta is >= 30deg
    allowance = 30.0
    heading_factor = 1.0 - heading_delta / allowance
    return max(heading_factor, 0.0)

def angle_of_vector(point1, point2):
    rad = math.atan2(point2[1] - point1[1], point2[0] - point1[0])
    return math.degrees(rad)


#===============================================================================
# TRACK POSITION
#===============================================================================

def race_line_distance_factor(params):
    global g_race_line_string

    # Reward for track position
    current_position = Point(params['x'], params['y'])
    distance = current_position.distance(g_race_line_string)

    # clamp reward to range (0..1) mapped to distance (track_width..0).
    # This could be negative since the car center can be off the track but
    # still not disqualified.
    allowance = params['track_width'] / 2.0 # gradient up to half width of track, then zero afterwards
    distance_factor = 1.0 - distance / allowance
    #print("x %0.2f y %0.2f distance %0.2f track_width %0.2f factor %0.7f" % (params['x'], params['y'], distance, params['track_width'], factor))
    return float(max(distance_factor, 0.0))


#===============================================================================
# EDGE FACTOR
#===============================================================================

def edge_factor(params):
    # Let the car at least get some small overhang, but a little more
    # feedback about proximity to edges rather than letting the car
    # the car just crash.  
    if params['distance_from_center'] >= (params['track_width']/2.0):
        edge_factor = 0.33
    elif not params['all_wheels_on_track']:
        # beginning of overhang, which is ok
        edge_factor = 0.67
    else:
        edge_factor = 1.0
    return edge_factor


#===============================================================================
# UTILS
#===============================================================================

def apply_weight(factor, weight):
    """Apply a weight to factor, clamping both arguments at 1.0
    Factor values will be 0..1. This function will cause the range of the
    factor values to be reduced according to:
      f = 1 - weight * (1 - factor)^easing
    In simple terms, a weight of 0.5 will cause the factor to only have weighted
    values of 0.5..1.0. If we further apply an easing, the decay from 1.0 toward
    the weighted minimum will be along a curve.
    """
    f_clamp = min(factor, 1.0)
    w_clamp = min(weight, 1.0)
    return 1.0 - w_clamp * (1.0 - f_clamp)
