from myo_raw import *
import pygame
import pandas as pd
import datetime
from copy import copy

DEBUG = False

class MyoRawHandler:
    def __init__(self, tty=None):
        self.m = MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
        self.create_connection()
        self.emg = []
        self.gyro = []
        self.acc = []
        self.quat = []

    def create_connection(self):
        self.m.add_emg_handler(self.proc_emg)
        self.m.add_imu_handler(self.proc_imu)
        self.m.connect()
        #self.m.add_arm_handler(lambda arm, xdir: print('arm', arm, 'xdir', xdir))
        #self.m.add_pose_handler(lambda p: print('pose', p))

    def test_hz(self, samples=100):
        start = time.time()
        for i in range(samples):
            self.m.run(1)
        elapsed_time = time.time() - start
        print("Elapsed time: {} | {} samples".format(elapsed_time, samples))

    def proc_emg(self, emg, moving, times=[]):
        if DEBUG:
            print("Current EMG: ", emg)
        self.emg = emg

    def proc_imu(self, quat, acc, gyro):
        if DEBUG:
            print("IMU data: ", quat, acc, gyro)
        self.gyro = gyro
        self.acc = acc
        self.quat = quat

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
        return {'emg': self.emg, 'gyro': self.gyro, 'acc': self.acc}

    def capture_movement(self, captures=400, description=""):
        self.tag = "myo"
        self.width = 200
        self.height = 200
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.all_data = pd.DataFrame(columns=["Timestamp", "Sensor 0", "Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4", "Sensor 5", "Sensor 6", "Sensor 7", "Gyro 0", "Gyro 1", "Gyro 2", "Acc 1", "Acc 2", "Acc 3"])

        while True:
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    self.all_data.to_csv("datasets/" + self.tag + '-' + description + "-" + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  + "-" + ".csv")
                    sys.exit()

                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_KP_ENTER:

                        print("----- Started movement capture -----")

                        emg_data = self.get_data()["emg"]

                        while(len(emg_data) != 8):
                            emg_data = self.get_data()["emg"]
                            if len(emg_data) >= 8:
                                break

                        batch_df = pd.DataFrame(columns=["Timestamp", "Sensor 0", "Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4", "Sensor 5", "Sensor 6", "Sensor 7", "Gyro 0", "Gyro 1", "Gyro 2", "Acc 1", "Acc 2", "Acc 3"])

                        for i in range(captures):
                            batch_data = self.get_data()
                            emg = [e for e in batch_data["emg"]]
                            gyro = [g for g in batch_data["gyro"]]
                            acc = [a for a in batch_data["acc"]]
                            all_data = emg + gyro + acc

                            if DEBUG:
                                print(all_data)

                            try:
                                batch_df = batch_df.append({'Timestamp': datetime.datetime.now().timestamp(), 'Sensor 0': all_data[0], 'Sensor 1': all_data[1], 'Sensor 2': all_data[2], 'Sensor 3': all_data[3], 'Sensor 4': all_data[4],'Sensor 5': all_data[5],'Sensor 6': all_data[6],'Sensor 7': all_data[7], 'Gyro 0': all_data[8], 'Gyro 1': all_data[9], 'Gyro 2': all_data[10], 'Acc 1': all_data[11], 'Acc 2': all_data[12], 'Acc 3': all_data[13]}, ignore_index=True)
                            except:
                                pass

                            print(batch_df.shape)

                        self.all_data = self.all_data.append(batch_df)
                        print("All data: ", self.all_data)
                        print("----- End of movement capture -----")
                    if event.key == pygame.K_c:
                        del self.all_data
                        self.tag = raw_input("Digite o novo tag do movimento")

def get_loop():
    while True:
        teste = myo.get_data()
        #print("Teste: ", teste)

myo = MyoRawHandler()
#myo.test_hz()
myo.capture_movement(captures=2000, description="parado-luiza")
#get_loop()
