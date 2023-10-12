import k3d


class skeleton:
    def __init__(self, joints, line_list):
        self.joints = joints
        self.lines = line_list

class SkeletonPainter:
    def __init__(self):
        super(SkeletonPainter).__init__()
        self.plot = k3d.plot(name='skeletool v0.1')
        self.JOINT_PAIRS = [('OP Neck', 'OP MidHip'), ('OP RHip', 'OP RKnee'), ('OP RKnee', 'OP RAnkle'), ('OP MidHip', 'OP RHip'), ('OP MidHip', 'OP LHip'), ('OP LHip', 'OP LKnee'),
              ('OP LKnee', 'OP LAnkle'), ('OP Neck', 'OP RShoulder'), ('OP RShoulder', 'OP RElbow'), ('OP RElbow', 'OP RWrist'), ('OP Neck', 'OP LShoulder'),
              ('OP LShoulder', 'OP LElbow'), ('OP LElbow', 'OP LWrist'), ('OP Neck', 'OP Nose'), ('OP Nose', 'OP REye'), ('OP Nose', 'OP LEye'),
              ('OP REye', 'OP REar'), ('OP LEye', 'OP LEar'), ('OP LAnkle', 'OP LBigToe'), ('OP RAnkle', 'OP RBigToe')]
        self.OP_21_JOINTS = self.OP_21_JOINTS = [
            'OP LWrist', 'OP RWrist', 'OP LAnkle', 'OP RAnkle',
            'OP LElbow', 'OP RElbow', 'OP LKnee', 'OP RKnee',
            'OP LShoulder', 'OP RShoulder',
            'OP LHip', 'OP MidHip', 'OP RHip',
            'OP Nose', 'OP Neck',
            'OP LEye', 'OP REye', 'OP LEar', 'OP REar',
            'OP LBigToe', 'OP RBigToe',
        ]

        self.JOINT_IDS = {self.OP_21_JOINTS[i]: i for i in range(len(self.OP_21_JOINTS))}

        self.JOINT_ID_PAIRS = [(self.JOINT_IDS[first], self.JOINT_IDS[second]) for first, second in self.JOINT_PAIRS]
        self.joint_list = []
        self.line_list = []

    def draw(self, joints3d, point_size=0.05, point_color=0xff):
        if len(joints3d) != 21:
            return False
        plt_points = k3d.points(positions=joints3d, point_size=point_size, shader='mesh', color=point_color)
        self.plot += plt_points
        self.joint_list.append(plt_points)
        lines = []
        for i, (first, second) in enumerate(self.JOINT_ID_PAIRS):
            if i == 9 or i == 12 or i == 6 or i == 2:
                plt_line = k3d.line(joints3d[[first, second], :], shader='mesh', color=0xff0000, width=point_size/5)
            else:
                plt_line = k3d.line(joints3d[[first, second], :], shader='mesh', color=0xffFFFF, width=point_size/5)

            self.plot += plt_line
            lines.append(plt_line)
        self.line_list.append(lines)

    def display(self):
        self.plot.display()

    def erase(self):
        self.plot -= self.joint_list[-1]
        for item in self.line_list[-1]:
            self.plot -= item
