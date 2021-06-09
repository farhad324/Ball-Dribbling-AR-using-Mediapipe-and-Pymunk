# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 19:55:05 2021

@author: Md Farhadul Islam
"""


import pymunk
import cv2
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

space = pymunk.Space()
space.gravity = 0, -100
choice=int(input("Enter '1' for 1 ball\nEnter '2' for 100 balls\n"))

if choice==1:
    number_of_balls=1
    balls_radius = 30
    elas=0.8
    gx0, gy0, gx00, gy00 = 0, 420,640, 420
    ground0 = pymunk.Segment(space.static_body, (gx0, gy0), (gx00, gy00), 10)    
    ground0.elasticity=elas
    space.add(pymunk.Body(body_type=pymunk.Body.STATIC), ground0)
    y_pos_diff=1
    hand_elas=0.3

else:
    number_of_balls=100
    balls_radius = 15
    y_pos_diff=30
    elas=0.5
    hand_elas=0.0
    
balls = [(300 + np.random.uniform(-100, 100), 250 + y_pos_diff*i + 0.5*i**2) for i in range(number_of_balls)]
balls_body = [pymunk.Body(100.0,1666, body_type=pymunk.Body.DYNAMIC) for b in balls]
for i, ball in enumerate(balls_body): 
    balls_body[i].position = balls[i]
    shape = pymunk.Circle(balls_body[i], balls_radius)
    shape.density=1
    shape.elasticity=elas
    space.add(balls_body[i], shape)


    
gx1, gy1, gx2, gy2 = 0, 60,640, 60
ground = pymunk.Segment(space.static_body, (gx1, gy1), (gx2, gy2), 10)    
ground.elasticity=elas
space.add(pymunk.Body(body_type=pymunk.Body.STATIC), ground)

gx3, gy3, gx4, gy4 = 0,60,0, 420
ground1 = pymunk.Segment(space.static_body, (gx3, gy3), (gx4, gy4), 10)
ground1.elasticity=elas
space.add(pymunk.Body(body_type=pymunk.Body.STATIC), ground1)

gx5, gy5, gx6, gy6 = 640,60,640, 420
ground2 = pymunk.Segment(space.static_body, (gx5, gy5), (gx6, gy6), 10)
ground2.elasticity=elas
space.add(pymunk.Body(body_type=pymunk.Body.STATIC), ground2)


fingers_radius = 20
fingers = [pymunk.Body(10,1666, body_type=pymunk.Body.KINEMATIC) for i in range(21)]
for i, finger in enumerate(fingers):
    finger_shape = pymunk.Circle(fingers[i], fingers_radius)
    finger_shape.elasticity=hand_elas
    space.add(fingers[i], finger_shape)


colors = [(219,152,52), (34, 126, 230),(60, 76, 231),(142, 59, 125),
          (113, 204, 46),(64, 73, 112), (15, 196, 241),(19,203,45)]


cap = cv2.VideoCapture(0) 
with mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        success, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if choice==1:
            cv2.line(image, (gx0, image.shape[0]-gy0), (gx00,image.shape[0]-gy00), (250, 100, 120), 10)
        cv2.line(image, (gx1, image.shape[0]-gy1), (gx2, image.shape[0]-gy2), (250, 100, 120), 10)
        cv2.line(image, (gx3, gy3), (gx4, gy4), (250, 100, 120), 10)
        cv2.line(image, (gx5, gy5), (gx6, gy6), (250, 100, 120), 10)
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(250, 44, 120), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121, 22, 176), thickness=2, circle_radius=2),
                                         )
                for i, finger in enumerate(fingers):

                    x = int(hand.landmark[i].x * image.shape[1])
                    y = image.shape[0]-int(hand.landmark[i].y * image.shape[0])

                    fingers[i].velocity = 8.0*(x - fingers[i].position[0]), 8.0*(y - fingers[i].position[1])


        for i, ball in enumerate(balls_body):
            xb = int(ball.position[0])
            yb = int(image.shape[0]-ball.position[1])
            cv2.circle(image, (xb, yb), balls_radius, colors[i%len(colors)], -1)
        space.step(0.02)
        
        cv2.imshow("Ball Dribbling", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()

