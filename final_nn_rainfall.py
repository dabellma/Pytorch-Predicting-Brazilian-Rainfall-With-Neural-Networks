import torch
import torch.distributed as dist
import os
import sys
import socket
import traceback
import datetime
import torch.nn as nn
import torch.optim as optim
from random import Random
import csv
import pandas as pd

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 50)
        self.fc4 = nn.Linear(50, 60)
        self.fc5 = nn.Linear(60, 70)
        self.fc6 = nn.Linear(70, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

"""Distributed Synchronous SGD Example"""
def run(rank, size):
    #torch.manual_seed(1234)

    with open('allRegionWithGHG.csv', newline='') as readfile:
        reader = csv.reader(readfile)
        dataFromCSV = tuple(reader)

    dataFrameFromCSV = pd.DataFrame(dataFromCSV, columns=['time_stamp',
    'dew_point_temperature',
    'wind_direction_meters_per_second',
    'radiation',
    'min_temperature_previous_hour',
    'atmospheric_pressure_at_station_height',
    'pressure_min_previous_hour',
    'max_dew_temperature_previous_hour',
    'wind_gust_meters_per_second',
    'air_temperature',
    'wind_speed',
    'max_temperature_previous_hour',
    'humidity_min_previous_hour',
    'pressure_max_previous_hour',
    'humidity_max_previous_hour',
    'air_relative_humidity_percentage',
    'min_dew_temperature_previous_hour',
    'total_precipitation',
    'co2',
    'cumulative_co2',
    'methane',
    'nitrous oxide'])


    dataFrameFromCSV = dataFrameFromCSV.drop(columns=['min_temperature_previous_hour', 'pressure_min_previous_hour', 'max_dew_temperature_previous_hour',
                    'max_temperature_previous_hour', 'humidity_min_previous_hour', 'pressure_max_previous_hour',
                    'humidity_max_previous_hour', 'min_dew_temperature_previous_hour'])

    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['dew_point_temperature'] != '']
    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['wind_direction_meters_per_second'] != '']
    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['radiation'] != '']
    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['atmospheric_pressure_at_station_height'] != '']
    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['wind_gust_meters_per_second'] != '']
    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['air_temperature'] != '']
    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['wind_speed'] != '']
    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['air_relative_humidity_percentage'] != '']
    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['total_precipitation'] != '']
    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['co2'] != '']
    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['cumulative_co2'] != '']
    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['methane'] != '']
    dataFrameFromCSV = dataFrameFromCSV[dataFrameFromCSV['nitrous oxide'] != '']

    dataFrameFromCSVAsTuples = list(dataFrameFromCSV.itertuples(index=False, name=None))
    dataFrameFromCSVAsTuplesNoTitles = dataFrameFromCSVAsTuples[1:]

#time stamp [0]
#dew point temperature [1]
#wind direction m/s [2]
#radiation [3]
#atmospheric pressure at station height [4]
#wind gust m/s [5]
#air temperature [6]
#wind speed [7]
#air relative humidity percentage [8]
#total precipitation [9]
#co2 [10]
#cumulative co2 [11]
#methane [12]
#nitrous oxide [13]

    featuresWithRainfallAtTheEndWithTimestamp = [(float(line[1]), float(line[2]),
               float(line[3]), float(line[4]), float(line[5]), float(line[6]),
               float(line[7]), float(line[8]), float(line[9]), float(line[10]),
               float(line[11]), float(line[12]), float(line[13])) for line in dataFrameFromCSVAsTuplesNoTitles]

#dew point temperature [0]
#wind direction m/s [1]
#radiation [2]
#atmospheric pressure at station height [3]
#wind gust m/s [4]
#air temperature [5]
#wind speed [6]
#air relative humidity percentage [7]
#total precipitation [8]
#co2 [9]
#cumulative co2 [10]
#methane [11]
#nitrous oxide [12]


    featuresWithRainfallAtTheEndTensors = [(torch.tensor((line[2], line[5], line[9])),
                torch.tensor(line[8]).reshape(1)) for line in featuresWithRainfallAtTheEndWithTimestamp]


    size = dist.get_world_size() + 1

    partition_sizes = [1.0 / size for _ in range(size)]
    partitionFromPartitioner = DataPartitioner(featuresWithRainfallAtTheEndTensors, partition_sizes)
    partition = partitionFromPartitioner.use(dist.get_rank())

    test_partition = partitionFromPartitioner.use(4)

    train_set = torch.utils.data.DataLoader(partition, shuffle=True)
    test_set = torch.utils.data.DataLoader(test_partition, shuffle=True)

    model = nn.parallel.DistributedDataParallel(Net()).float()
        # model = load_model(nn.parallel.DistributedDataParallel(Net()), "best_model.pth").float()

    optimizer = optim.SGD(model.parameters(), lr=1E-9, momentum=0.5)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    for epoch in range(20):
        epoch_loss = 0.0
        for i, (data, target) in enumerate(train_set):
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        if dist.get_rank() == 0:
            print(epoch_loss)
        if dist.get_rank() == 0 and epoch_loss < best_loss:
            best_loss = epoch_loss

    print("Done training, now predictions")
    if (dist.get_rank() == 0):
            predictionCumulative = 0
            actualCumulative = 0
            print("Predicted Precipitation (mm)                    Actual Precipitation (mm)")
            for i, (data, target) in enumerate(test_set):
                print(model(data)[0][0], "     ",  target[0][0])
                predictionCumulative += model(data)[0][0]
                actualCumulative += target[0][0]
            print(predictionCumulative)
            print(actualCumulative)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'cheyenne'
    os.environ['MASTER_PORT'] = '30915'

    #initialize the process group
    dist.init_process_group("gloo", rank=int(rank), world_size=int(world_size), init_method='tcp://cheyenne:23456', timeout=datetime.timedelta(weeks=120))


if __name__ == "__main__":
    try:
        setup(sys.argv[1], sys.argv[2])
        print(socket.gethostname() + ": Setup completed!")
        run(int(sys.argv[1]), int(sys.argv[2]))
    except Exception as e:
        traceback.print_exc()
        sys.exit(3)