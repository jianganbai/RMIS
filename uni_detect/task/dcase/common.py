DCASE20_MT = {
    'dev': ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve'],
    'eval': ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
}
DCASE21_MT = {
    'dev': ['fan', 'gearbox', 'pump', 'slider', 'ToyCar', 'ToyTrain', 'valve'],
    'eval': ['fan', 'gearbox', 'pump', 'slider', 'ToyCar', 'ToyTrain', 'valve']
}
DCASE22_MT = {
    'dev': ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'],
    'eval': ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve']
}
DCASE23_MT = {
    'dev': ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'],
    'eval': ['bandsaw', 'grinder', 'shaker', 'ToyDrone', 'ToyNscale', 'ToyTank', 'Vacuum']
}
DCASE24_MT = {
    'dev': ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'],
    'eval': [
        '3DPrinter', 'AirCompressor', 'BrushlessMotor', 'HairDryer', 'HoveringDrone',
        'RoboticArm', 'Scanner', 'ToothBrush', 'ToyCircuit'
    ]
}

DCASE25_MT = {
    'dev': ['bearing', 'fan', 'gearbox', 'slider', 'ToyCar', 'ToyTrain', 'valve'],
    'eval': [
        'AutoTrash', 'BandSealer', 'CoffeeGrinder', 'HomeCamera',
        'Polisher', 'ScrewFeeder', 'ToyPet', 'ToyRCCar'
    ]
}

SET_MT_MAP = {
    'dcase20': DCASE20_MT,
    'dcase21': DCASE21_MT,
    'dcase22': DCASE22_MT,
    'dcase23': DCASE23_MT,
    'dcase24': DCASE24_MT,
    'dcase25': DCASE25_MT,
}

DCASE20_MTSEC = {
    'dev': {
        'fan': [0, 2, 4, 6],
        'pump': [0, 2, 4, 6],
        'slider': [0, 2, 4, 6],
        'ToyCar': [1, 2, 3, 4],
        'ToyConveyor': [1, 2, 3],
        'valve': [0, 2, 4, 6]
    },
    'eval': {
        'fan': [1, 3, 5],
        'pump': [1, 3, 5],
        'slider': [1, 3, 5],
        'ToyCar': [5, 6, 7],
        'ToyConveyor': [4, 5, 6],
        'valve': [1, 3, 5]
    }
}

DCASE21_MTSEC = {
    'dev': {
        'fan': [0, 1, 2],
        'gearbox': [0, 1, 2],
        'pump': [0, 1, 2],
        'slider': [0, 1, 2],
        'ToyCar': [0, 1, 2],
        'ToyTrain': [0, 1, 2],
        'valve': [0, 1, 2]
    },
    'eval': {
        'fan': [3, 4, 5],
        'gearbox': [3, 4, 5],
        'pump': [3, 4, 5],
        'slider': [3, 4, 5],
        'ToyCar': [3, 4, 5],
        'ToyTrain': [3, 4, 5],
        'valve': [3, 4, 5]
    }
}

DCASE22_MTSEC = {
    'dev': {
        'bearing': [0, 1, 2],
        'fan': [0, 1, 2],
        'gearbox': [0, 1, 2],
        'slider': [0, 1, 2],
        'ToyCar': [0, 1, 2],
        'ToyTrain': [0, 1, 2],
        'valve': [0, 1, 2]
    },
    'eval': {
        'bearing': [3, 4, 5],
        'fan': [3, 4, 5],
        'gearbox': [3, 4, 5],
        'slider': [3, 4, 5],
        'ToyCar': [3, 4, 5],
        'ToyTrain': [3, 4, 5],
        'valve': [3, 4, 5]
    }
}

DCASE23_MTSEC = {
    'dev': {
        'bearing': [0],
        'fan': [0],
        'gearbox': [0],
        'slider': [0],
        'ToyCar': [0],
        'ToyTrain': [0],
        'valve': [0]
    },
    'eval': {
        'bandsaw': [0],
        'grinder': [0],
        'shaker': [0],
        'ToyDrone': [0],
        'ToyNscale': [0],
        'ToyTank': [0],
        'Vacuum': [0]
    }
}

DCASE24_MTSEC = {
    'dev': {
        'bearing': [0],
        'fan': [0],
        'gearbox': [0],
        'slider': [0],
        'ToyCar': [0],
        'ToyTrain': [0],
        'valve': [0]
    },
    'eval': {
        '3DPrinter': [0],
        'AirCompressor': [0],
        'BrushlessMotor': [0],
        'HairDryer': [0],
        'HoveringDrone': [0],
        'RoboticArm': [0],
        'Scanner': [0],
        'ToothBrush': [0],
        'ToyCircuit': [0]
    }
}

DCASE25_MTSEC = {
    'dev': {
        'bearing': [0],
        'fan': [0],
        'gearbox': [0],
        'slider': [0],
        'ToyCar': [0],
        'ToyTrain': [0],
        'valve': [0]
    },
    'eval': {
        'AutoTrash': [0],
        'BandSealer': [0],
        'CoffeeGrinder': [0],
        'HomeCamera': [0],
        'Polisher': [0],
        'ScrewFeeder': [0],
        'ToyPet': [0],
        'ToyRCCar': [0]
    }
}

ALL_MTSEC = {
    'dcase20': DCASE20_MTSEC,
    'dcase21': DCASE21_MTSEC,
    'dcase22': DCASE22_MTSEC,
    'dcase23': DCASE23_MTSEC,
    'dcase24': DCASE24_MTSEC,
    'dcase25': DCASE25_MTSEC,
}
