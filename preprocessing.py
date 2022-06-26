"""
Eye Tracking: Experiment design and machine learning methods
Assignment 3

Authors: Cui Ding(olatname: cding)
Matriculation Numbers: 21-718-945
"""


from argparse import ArgumentParser, FileType
from typing import TextIO, List, Tuple, Dict
import csv
from statistics import mean
from math import sqrt
import matplotlib.pyplot as plt


def read_file(infile: TextIO, trial_id: int):
    """
    Read in the raw eye tracking data
    :param infile: raw eye tracking file: trialID, pointID, time, x_left, y_left, pupil_left, x_right, y_right, pupil_right
    :param trial_id: the trial ID to focus on
    :return:
    """
    with open(infile, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        gaze_data = [{'t':int(row['time']), 'x':float(row['x_right']), 'y':float(row['y_right'])} for row in csvreader \
        if int(row['trialId']) == trial_id and row['x_right'] and row['y_right']]
    # print(gaze_data)
    return gaze_data


def ivt(gaze_data: List[Tuple[str]], vel_thres: float, dur_thres: float, freq: int) -> List[Dict]:
    """ Implementation of the velocity-based fixation detection algorithm. """
    fixations = []
    velocity_data =[]
    for i in range(1, len(gaze_data)):
        distance = sqrt((gaze_data[i]['x']-gaze_data[i-1]['x'])**2 + (gaze_data[i]['y']-gaze_data[i-1]['y'])**2)
        velocities = distance * freq / 1000
        if velocities <= vel_thres:
            velocity_data.append((i, velocities))
            if i == len(gaze_data)-1 and len(velocity_data) > (dur_thres*freq/1000):
                j = velocity_data[0][0]-1
                start_t = gaze_data[j]['t']
                end_t = gaze_data[i]['t']
                x_mean = mean([gaze_data[indx]['x'] for indx in range(j, i)])
                y_mean = mean([gaze_data[indx]['y'] for indx in range(j, i)])
                duration = end_t - start_t
                centroid = {'x_mean':x_mean,
                    'y_mean':y_mean,
                    'start_t':start_t,
                    'end_t':end_t,
                    'duration': duration}
                fixations.append(centroid)
        elif len(velocity_data) > (dur_thres*freq/1000):
#         elif len(velocity_data) > 1 and gaze_data[i-1]['t'] - gaze_data[velocity_data[0][0]-1]['t'] > dur_thres:
            j = velocity_data[0][0]-1
            start_t = gaze_data[j]['t']
            end_t = gaze_data[i-1]['t']
            x_mean = mean([gaze_data[indx]['x'] for indx in range(j, i)])
            y_mean = mean([gaze_data[indx]['y'] for indx in range(j, i)])
            duration = end_t - start_t
            centroid = {'x_mean':x_mean,
                'y_mean':y_mean,
                'start_t':start_t,
                'end_t':end_t,
                'duration': duration}
            fixations.append(centroid)
            velocity_data.clear()
        else:
            velocity_data.clear()
            
### Solution_2: not done by me

#     fixations = []
#     while gaze_data and gaze_data[-1]['t'] - gaze_data[0]['t'] > dur_thres:
#         window = [p for p in gaze_data if p['t'] - gaze_data[0]['t'] <= dur_thres]
#         window.append(gaze_data[len(window)])
#         if dispersion(window) <= dis_thres:
#             while dispersion(window) <= dis_thres and len(window) < len(gaze_data):
#                 window.append(gaze_data[len(window)])
#             window.pop(-1)
#             centroid = {'x_mean':mean([p['x'] for p in window]),
#                    'y_mean':mean([p['y'] for p in window]),
#                    'start_t':window[0]['t'],
#                    'end_t':window[-1]['t'],
#                    'duration': window[-1]['t'] - window[0]['t']}
#             fixations.append(centroid)
#             gaze_data = gaze_data[len(window):]
#         else:
#             gaze_data = gaze_data[1:]

    return fixations
            

def compute_dispersion(window : list) -> float:
    x_coordinates = [p['x'] for p in window]
    y_coordinates = [p['y'] for p in window]
    x_range = max(x_coordinates) -  min(x_coordinates)
    y_range = max(y_coordinates) -  min(y_coordinates)
    return 0.5*(x_range + y_range)


def idt(gaze_data: List[Tuple[str]], dis_thres: float, dur_thres: float) -> List[Dict]:
    """ Implementation of the dispersion-based fixation algorithm. """
    fixations = []
    while gaze_data and gaze_data[-1]['t'] - gaze_data[0]['t'] > dur_thres:
        window = [p for p in gaze_data if p['t'] - gaze_data[0]['t'] <= dur_thres]
        window.append(gaze_data[len(window)])
        if compute_dispersion(window) <= dis_thres:
            while compute_dispersion(window) <= dis_thres and len(window) < len(gaze_data):
                window.append(gaze_data[len(window)])
            window.pop(-1)
            centroid = {'x_mean':mean([p['x'] for p in window]),
                   'y_mean':mean([p['y'] for p in window]),
                   'start_t':window[0]['t'],
                   'end_t':window[-1]['t'],
                   'duration': window[-1]['t'] - window[0]['t']}
            fixations.append(centroid)
            gaze_data = gaze_data[len(window):]
        else:
            gaze_data = gaze_data[1:]
    return fixations
    
    
def visualise(gaze_data, fixations, args):
    """
    Too tired to write anything.
    """
    time = []
    x = []
    y = []
    for p in gaze_data:
        time.append(p['t'])
        x.append(p['x'])
        y.append(p['y'])

    plt.figure()
    plt.plot(time, x)
    plt.plot(time, y)
    for f in fixations:
        plt.axvspan(f['start_t'], f['end_t'], color = 'silver')
    plt.xlabel('time(ms)')
    plt.ylabel('postion in pixels')
    plt.legend(['horizontal movement', 'vertical movement','fixation'])
    if args.mode == 'dispersion':
        plt.title(f"{args.freq} Hz, Trial {args.trial} ({args.mode}, {args.dis_thres} pixels, {args.dur_thres} ms)")
    else:
        plt.title(f"{args.freq} Hz, Trial {args.trial} ({args.mode}, {args.vel_thres} pixels/ms, {args.dur_thres} ms)")
    plt.show()
    return


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Preprocess raw eye tracking data.')
    parser.add_argument('infile', type=FileType('r', encoding='utf-8'),
                        help='Input file with raw eye tracking data.')
    parser.add_argument('--mode', type=str, choices=['dispersion', 'velocity'], required=True,
                        help='The kind of algorithm to apply for fixation detection')
    parser.add_argument('--trial', type=int, help='the trial ID')
    parser.add_argument('--freq', type=int, choices=[60, 2000], required=True,
                        help='the sampling frequency')
    parser.add_argument('--vel_thres', type=float, help='the min velocity threshold for I-VT, in pixels/ms')
    parser.add_argument('--dis_thres', type=float, help='the max dispersion threshold for I-DT, in pixels')
    parser.add_argument('--dur_thres', type=float, default=200, help='the min duration threshold for I-DT in ms')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
#     read file according to --freq and --trial
    gaze_data = read_file("JumpingDots"+str(args.freq)+".csv", args.trial)
    if args.mode == 'dispersion':
        fixations = idt(gaze_data, args.dis_thres, args.dur_thres)
        for f in fixations:
            print(f)
    else:
        fixations = ivt(gaze_data, args.vel_thres, args.dur_thres, args.freq)
        for f in fixations:
            print(f)
            
    visualise(gaze_data, fixations, args)

if __name__ == '__main__':
    main()
