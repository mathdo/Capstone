#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from copy import deepcopy
from std_msgs.msg import Int32

import math
from tf.transformations import euler_from_quaternion
'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
G = 9.8
MU = 0.7
TPRT = 0.5
SAFE_DIST = 8.0

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb, queue_size=1)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)


        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=1)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.position = None    # current position (x, y, z) of ego vehicle
        self.yaw = None         # current yaw of ego vehicle

        self.curr_vel = 0

        self.base_waypoints = []
        self.final_waypoints = []

        self.traffic_light_idx = -1
        self.speed_limit = 11.1


        rospy.spin()

    def pose_cb(self, msg):
        #rospy.loginfo("In pose callback")
        self.position = msg.pose.position
        orientation = msg.pose.orientation
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        self.yaw = yaw
        #rospy.loginfo("Pose callback - got position " + str(msg.pose.position) + ", orientation " + str(
        #    msg.pose.orientation) + ", computed yaw: " + str(self.yaw))

        if len(self.base_waypoints) > 0:
            self.compute_next_waypoints(slow_acceleration=True)
            lane = Lane()
            lane.header.frame_id = '/world'
            lane.header.stamp = rospy.Time.now()
            lane.waypoints = self.final_waypoints
            #ospy.loginfo("final_waypoints: " + str(self.final_waypoints))
            self.final_waypoints_pub.publish(lane)

    def velocity_cb(self, msg):
        self.curr_vel = msg.twist.linear.x

    def compute_next_waypoints(self, slow_acceleration=False):
        # compute and publish next waypoints
        next_waypoint_idx = self.get_next_waypoint()
        # rospy.loginfo("next_waypoint_idx = " + str(next_waypoint_idx))
        del self.final_waypoints[:]
        for i in range(LOOKAHEAD_WPS):
            wp = deepcopy(self.base_waypoints[(next_waypoint_idx + i) % len(self.base_waypoints)])
            if slow_acceleration:
                wp.twist.twist.linear.x = min((self.curr_vel + (i+1)*0.5), self.speed_limit)
            self.final_waypoints.append((wp))


    def waypoints_cb(self, lane):
        #rospy.loginfo("In waypoints callback")
        if len(self.base_waypoints) == 0 and lane is not None and len(lane.waypoints) > 0:
            #rospy.loginfo("In waypoints callback, passed if.")
            self.base_waypoints = lane.waypoints

        #rospy.loginfo("waypoints length = " + str(len(self.base_waypoints)))
        #rospy.loginfo("waypoints[0].pose.pose.position: " + str(self.base_waypoints[0].pose.pose.position))
        #rospy.loginfo("waypoints[0].pose.pose.orientation: " + str(self.base_waypoints[0].pose.pose.orientation))

    # Get closest waypoint index(from P11 - Path planning project)
    def get_closest_waypoint_idx(self):

        closest_distance_found = 10e9
        closest_index = -1
        waypoints = self.base_waypoints

        for i in range(len(waypoints)):
            p2 = waypoints[i].pose.pose.position
            curr_dist = self.euc_dist(self.position, p2)
            if curr_dist < closest_distance_found:
                closest_distance_found = curr_dist
                closest_index = i

        return closest_index

    # Get next waypoint index(from P11 - Path planning project)
    def get_next_waypoint(self):

        next_index = self.get_closest_waypoint_idx()
        #rospy.loginfo("get_closest_waypoint_idx returned: " + str(next_index))

        p1 = self.position
        p2 = self.base_waypoints[next_index].pose.pose.position

        direction = math.atan2((p2.y - p1.y), (p2.x - p1.x))

        angle = abs(self.yaw - direction)

        if angle > math.pi / 4:
            next_index += 1

        return next_index % len(self.base_waypoints)

    def traffic_cb(self, msg):
        if len(self.base_waypoints) == 0:
            return
        next_index = self.get_closest_waypoint_idx()
        next_red_tl_index = msg.data
        if next_red_tl_index < 0:
            self.clear_waypoint_velocities()
            self.compute_next_waypoints(slow_acceleration=True)
            return

        if self.traffic_light_idx !=  next_red_tl_index:
            self.clear_waypoint_velocities()
            self.traffic_light_idx = next_red_tl_index
            for wp_index in range(next_index, next_red_tl_index + 1):
                dist = self.distance(self.base_waypoints, wp_index, next_red_tl_index) - SAFE_DIST
                if dist < 0.0:
                    safe_speed = 0.0
                else:
                    safe_speed = self.calc_safe_speed(dist)
                self.set_waypoint_velocity(self.base_waypoints, wp_index, min(safe_speed, self.speed_limit))
            self.compute_next_waypoints(slow_acceleration=True)

    def clear_waypoint_velocities(self):
        for i in range(len(self.base_waypoints)):
            self.set_waypoint_velocity(self.base_waypoints, i, self.speed_limit)
        for i in range(len(self.final_waypoints)):
            self.set_waypoint_velocity(self.final_waypoints, i, self.speed_limit)
        # rospy.loginfo("Reset all velocities to speed limit.")

    # dist is in meters, returning safe speed in m/s, formula according to ACDA
    def calc_safe_speed(self, dist):
        vel = math.sqrt( (G * G * MU * MU * TPRT * TPRT) + (2.0 * MU * G * dist) ) - (MU * G * TPRT)
        return max(vel, 0.0)


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def euc_dist(self, p1, p2):
        return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2)

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
