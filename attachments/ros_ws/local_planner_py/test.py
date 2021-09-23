#! /usr/bin/env python2

global_step = 0

def update(frame_num):
    global global_step
    global_step = frame_num

def main():
    global_step = 0
    update(2)

if __name__ == "__main__":
    main()