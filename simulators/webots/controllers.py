
class WebotsController:
    def __init__(self, robot):
        self.robot = robot

        motors 
        for i in range(robot.getNumberOfDevices()):
            device = robot.getDeviceByIndex(i)
            ty = device.getNodeType()
            if ty == device.ROTATIONAL_MOTOR:


    def step(self):
        pass

    def powerConsumption(self):
        return 0
