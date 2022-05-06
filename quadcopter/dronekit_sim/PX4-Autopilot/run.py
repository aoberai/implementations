
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import NavSatFix
from geographic_msgs.msg import GeoPoseStamped
from mavros_msgs.msg import HomePosition

RATE = 5 # Hz
current_global_pos = NavSatFix()
current_home_pos = HomePosition()

def _global_pos_cb(msg):
    global current_global_pos
    current_global_pos = msg

def _home_pos_cb(msg):
    global current_home_pos
    current_home_pos = msg


global_pos_sub = rospy.Subscriber('/mavros/global_position/global', NavSatFix, _global_pos_cb)
home_pos_sub = rospy.Subscriber('/mavros/home_position/home', HomePosition, _home_pos_cb)
pos_pub = rospy.Publisher("/mavros/setpoint_position/global", GeoPoseStamped, queue_size=1)


def main():
    rospy.init_node('setpoint_test', anonymous=False)
    rate = rospy.Rate(RATE)

    while True:
        msg = GeoPoseStamped()
        header = Header()
        header.stamp = rospy.Time.now()
        msg.header = header
        msg.pose.position.latitude = current_global_pos.latitude
        msg.pose.position.longitude = current_global_pos.longitude
        msg.pose.position.altitude = current_global_pos.altitude
        # Uncomment to go to the HomePosition and keep the current altitude
        # msg.pose.position = current_home_pos.geo

        msg.pose.orientation = current_home_pos.orientation
        pos_pub.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    main()
