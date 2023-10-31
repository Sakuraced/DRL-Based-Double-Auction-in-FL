import torch as T
import torch.nn as nn
import torch.optim as optim

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorNetwork, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(
                in_features=state_dim,  # 第一个隐藏层的输入，数据的特征数
                out_features=fc1_dim,  # 第一个隐藏层的输出，神经元的数量
                bias=True
            ),
            nn.Sigmoid()
        )
        # 定义第二个隐藏层
        self.hidden2 = nn.Sequential(
            nn.Linear(fc1_dim, fc2_dim),
            nn.Sigmoid()
        )
        # 分类层
        self.classify = nn.Sequential(
            nn.Linear(fc2_dim, action_dim),
            nn.Softmax(dim=1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.classify(fc2)
        # 输出为两个隐藏层和两个输入层
        return output

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=-1)
        x = T.relu(self.ln1(self.fc1(x)))
        x = T.relu(self.ln2(self.fc2(x)))
        q = self.q(x)

        return q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))