from myo_raw import *


class MyoRawHandler:
    def __init__(self, tty=None):
        self.m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
        self.create_connection()

    def create_connection(self):
       self.m.add_emg_handler(self.proc_emg)
       self.m.connect()
       self.m.add_arm_handler(lambda arm, xdir: print('arm', arm, 'xdir', xdir))
       self.m.add_pose_handler(lambda p: print('pose', p))

    def test_hz(self, samples=100):
        start = time.time()
        for i in range(samples):
            self.m.run(1)
        elapsed_time = time.time() - start
        print("Elapsed time: {} | {} samples".format(elapsed_time, samples))

    def proc_emg(self, emg, moving, times=[]):
        print(emg)
        times.append(time.time()) # timestamp
        if len(times) > 20: # only keeps 20 timestamps in memory
            print((len(times) - 1) / (times[-1] - times[0]))
            times.pop(0) # removes first timestamp in list

    def get_data(self):
        """
        run calls recv_packet function from BT class
        -> recv_packet function:
            receives byte message from serial (which is myo's dongle)
            the message is treated by BT's function proc_byte
            -> proc_byte function:
                add first few bytes to buffer to calculate the packet length
                then, reads message of calculated package length and deletes the buffer
                returns a Packet class
            the returned packet has a 'typ' section, which has to be equal to 0x80
            if that conditions is true, the packet is sent to handle_event function
            -> handle_event function:

        """
        self.m.run(1)

myo = MyoRawHandler()
myo.test_hz()
#while True:
#    myo.get_data()

