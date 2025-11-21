import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# 1. 장치 설정 (M1 MPS 사용)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# 2. 데이터 전처리 및 증강 설정
# 계획서의 지침에 따라 CIFAR-10 전용 평균/표준편차와 증강 기법을 적용합니다.
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 데이터 증강: 이미지를 랜덤하게 자름
    transforms.RandomHorizontalFlip(),         # 데이터 증강: 좌우 반전
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # 정규화
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 3. 데이터셋 다운로드 및 로더 생성 (배치 크기 128)
batch_size = 128 

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

print("데이터 준비 완료!")