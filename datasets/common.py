# DCASE20
MT_MAP20 = {
    'fan': 0,
    'pump': 1,
    'slider': 2,
    'ToyCar': 3,
    'ToyConveyor': 4,
    'valve': 5
}
INV_MT_MAP20 = {v: k for k, v in MT_MAP20.items()}


# DCASE21
MT_MAP21 = {
    'fan': 0,
    'gearbox': 1,
    'pump': 2,
    'slider': 3,
    'ToyCar': 4,
    'ToyTrain': 5,
    'valve': 6
}
INV_MT_MAP21 = {v: k for k, v in MT_MAP21.items()}


# DCASE22
MT_MAP22 = {
    'bearing': 0,
    'fan': 1,
    'gearbox': 2,
    'slider': 3,
    'ToyCar': 4,
    'ToyTrain': 5,
    'valve': 6,
}
INV_MT_MAP22 = {v: k for k, v in MT_MAP22.items()}


# DCASE23
MT_MAP23 = {
    'bearing': 0,
    'fan': 1,
    'gearbox': 2,
    'slider': 3,
    'ToyCar': 4,
    'ToyTrain': 5,
    'valve': 6,
    'bandsaw': 7,
    'grinder': 8,
    'shaker': 9,
    'ToyDrone': 10,
    'ToyNscale': 11,
    'ToyTank': 12,
    'Vacuum': 13
}
INV_MT_MAP23 = {v: k for k, v in MT_MAP23.items()}


# DCASE24
MT_MAP24 = {
    'bearing': 0,
    'fan': 1,
    'gearbox': 2,
    'slider': 3,
    'ToyCar': 4,
    'ToyTrain': 5,
    'valve': 6,
    '3DPrinter': 7,
    'AirCompressor': 8,
    'BrushlessMotor': 9,
    'HairDryer': 10,
    'HoveringDrone': 11,
    'RoboticArm': 12,
    'Scanner': 13,
    'ToothBrush': 14,
    'ToyCircuit': 15
}
INV_MT_MAP24 = {v: k for k, v in MT_MAP24.items()}

# DCASE25
MT_MAP25 = {
    'bearing': 0,
    'fan': 1,
    'gearbox': 2,
    'slider': 3,
    'ToyCar': 4,
    'ToyTrain': 5,
    'valve': 6,
    'AutoTrash': 7,
    'BandSealer': 8,
    'CoffeeGrinder': 9,
    'HomeCamera': 10,
    'Polisher': 11,
    'ScrewFeeder': 12,
    'ToyPet': 13,
    'ToyRCCar': 14
}
INV_MT_MAP25 = {v: k for k, v in MT_MAP25.items()}


ALL_MT_MAP = {
    'dcase20': MT_MAP20,
    'dcase21': MT_MAP21,
    'dcase22': MT_MAP22,
    'dcase23': MT_MAP23,
    'dcase24': MT_MAP24,
    'dcase25': MT_MAP25,
}
