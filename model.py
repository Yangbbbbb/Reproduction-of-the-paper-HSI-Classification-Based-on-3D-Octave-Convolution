import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class HSINet(nn.Module):
    def __init__(self, keep_pro=0.5, band=103):
        super(HSINet, self).__init__()
        self.keep_prob = keep_pro
        self.band = band

        self.batch_normal3d = nn.BatchNorm3d

        # 定义octconv_1卷积层和池化层
        self.conv_hh_1 = nn.Conv3d(1, 24, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))
        self.maxpool_1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv_hl_1 = nn.Conv3d(1, 24, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))


        # 定义octconv_2卷积层和池化层
        self.conv_hh_2 = nn.Conv3d(24, 48, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))
        self.maxpool_2 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv_hl_2 = nn.Conv3d(24, 48, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))
        self.conv_lh_2 = nn.Conv3d(24, 48, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))
        self.conv_ll_2 = nn.Conv3d(24, 48, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))

        # 定义octconv_3卷积层和池化层
        self.conv_hh_3 = nn.Conv3d(48, 24, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))
        self.avgpool_3 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv_hl_3 = nn.Conv3d(48, 24, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))

        # 定义octconv_4卷积层和池化层
        self.conv_hh_4 = nn.Conv3d(24, 1, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))
        self.avgpool_4 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv_hl_4 = nn.Conv3d(24, 1, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))
        self.conv_lh_4 = nn.Conv3d(24, 1, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))
        self.conv_ll_4 = nn.Conv3d(24, 1, kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))

        # res_block
        self.batch_normal_res_1 = nn.BatchNorm3d(24)
        self.batch_normal_res_2 = nn.BatchNorm3d(24)
        self.relu = nn.ReLU()
        self.batch_normal_res_3 = nn.BatchNorm3d(48)
        self.batch_normal_res_4 = nn.BatchNorm3d(48)
        self.maxpool_res=nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.batch_normal_res_5 = nn.BatchNorm3d(24)
        self.batch_normal_res_6 = nn.BatchNorm3d(24)
        self.batch_normal_res_7 = nn.BatchNorm3d(1)
        self.batch_normal_res_8 = nn.BatchNorm3d(1)

        # Spatial Attention
        self.spatial_convA = nn.Conv2d(self.band, self.band, kernel_size=3, stride=1, padding=1)
        self.spatial_bnA = nn.BatchNorm2d(self.band)
        self.spatial_conv = nn.Conv2d(self.band, self.band, kernel_size=1, stride=1)
        self.spatial_bn = nn.BatchNorm2d(self.band)
        self.softmax=nn.Softmax()

        # Spectral Attention
        self.spectral_biases = nn.Conv2d(self.band, self.band, kernel_size=1, stride=1)
        self.spectral_bn = nn.BatchNorm2d(self.band)

        # logits_spatial
        self.f1_spa=nn.Linear(self.band * 7 * 7, 1024)
        self.dropout_spa=nn.Dropout(p=1 - self.keep_prob)
        self.f2_spa =  nn.Linear(1024, 9)

        # logits_spectralctral
        self.f1_spe = nn.Linear(self.band * 7 * 7, 1024)
        self.dropout_spe = nn.Dropout(p=1 - self.keep_prob)
        self.f2_spe = nn.Linear(1024, 9)

        # classer
        self.f1_classer = nn.Linear(self.band * 7 * 7, 1024)
        self.dropout_classe = nn.Dropout(p=1 - self.keep_prob)
        self.f2_classer = nn.Linear(1024, 9)



    def octconv_1(self, input_high):
        data_dim = torch.reshape(input_high, [-1, 13, 13, self.band, 1])
        aa = data_dim.permute(0, 4, 3, 1, 2)
        data_hh = self.conv_hh_1(aa)
        high_to_low = self.maxpool_1(aa)
        data_hl = self.conv_hl_1(high_to_low)
        return data_hh, data_hl

    def octconv_2(self, input_high, input_low):
        data_hh = self.conv_hh_2(input_high)
        data_down = self.maxpool_2(input_high)
        data_hl = self.conv_hl_2(data_down)

        data_lh_conv = self.conv_lh_2(input_low)
        aa = data_lh_conv.permute(0, 3, 4, 1, 2 )
        bb = torch.reshape(aa, [-1, 7, 7, 48 * self.band]).permute(0, 3, 1, 2)
        data_lh = F.interpolate(bb, size=(13, 13), mode='bilinear', align_corners=False)
        cc = torch.reshape(data_lh.permute(0, 2, 3, 1), [-1, 13, 13, 48,self.band])
        dd = cc.permute(0, 3, 4, 1, 2)

        data_ll = self.conv_ll_2(input_low)
        data_high = data_hh + dd

        data_low = data_hl + data_ll
        return data_high, data_low

    def octconv_3(self, input_high):
        data_hh = self.conv_hh_3(input_high)
        high_to_low = self.avgpool_3(input_high)
        data_hl = self.conv_hl_3(high_to_low)
        return data_hh, data_hl

    def octconv_4(self, input_high, input_low):
        data_hh = self.conv_hh_4(input_high)
        data_down = self.avgpool_4(input_high)
        data_hl = self.conv_hl_4(data_down)

        data_lh_conv = self.conv_lh_4(input_low)
        aa = data_lh_conv.permute(0, 3, 4, 1, 2)
        bb = torch.reshape(aa, [-1, 4, 4, 1 * self.band]).permute(0, 3, 1, 2)
        data_lh = F.interpolate(bb, size=(7, 7), mode='bilinear', align_corners=False)
        cc = torch.reshape(data_lh.permute(0, 2, 3, 1), [-1, 7, 7, 1, self.band])
        dd = cc.permute(0, 3, 4, 1, 2)

        data_ll = self.conv_ll_4(input_low)
        data_high = data_hh + dd
        data_low = data_hl + data_ll
        return data_high, data_low

    def res_block(self, input):
        octconv1_high, octconv1_low = self.octconv_1(input)
        bn1_high = self.batch_normal_res_1(octconv1_high)
        bn1_low = self.batch_normal_res_2(octconv1_low)
        relu1_high = self.relu(bn1_high)
        relu1_low = self.relu(bn1_low)

        octconv2_high, octconv2_low = self.octconv_2(relu1_high, relu1_low)
        bn2_high = self.batch_normal_res_3(octconv2_high)
        bn2_low = self.batch_normal_res_4(octconv2_low)
        relu2_high = self.relu(bn2_high)
        relu2_low = self.relu(bn2_low)

        pool_high = self.maxpool_res(relu2_high)
        data_fusion1 = pool_high + relu2_low

        octconv3_high, octconv3_low = self.octconv_3(data_fusion1)
        bn3_high =self.batch_normal_res_5(octconv3_high)
        bn3_low = self.batch_normal_res_6(octconv3_low)
        relu3_high = self.relu(bn3_high)
        relu3_low = self.relu(bn3_low)

        octconv4_high, octconv4_low = self.octconv_4(relu3_high, relu3_low)
        bn4_high = self.batch_normal_res_7(octconv4_high)
        bn4_low = self.batch_normal_res_8(octconv4_low)
        relu4_high = self.relu(bn4_high)
        relu4_low = self.relu(bn4_low)

        aa = relu4_high.permute(0, 3, 4, 1, 2)
        bb = torch.reshape(aa, [-1, 7, 7, 1 * self.band]).permute(0, 3, 1, 2)

        cc = relu4_low.permute(0, 3, 4, 1, 2)
        dd = torch.reshape(cc, [-1, 4, 4, self.band]).permute(0, 3, 1, 2)

        relu4_low_up = F.interpolate(input=dd, size=[7, 7], mode='bilinear', align_corners=False)
        out = bb + relu4_low_up
        return out

    def spatial_attention(self, input_spatial):
        spatial_convA = self.spatial_convA(input_spatial)
        spatial_bnA = self.spatial_bnA (spatial_convA)
        spatial_reluA = self.relu(spatial_bnA)

        spatialA_reshape_1 = torch.reshape(spatial_reluA, [-1, self.band, 7 * 7])
        spatialA_transpose = spatialA_reshape_1.permute(0, 2, 1)
        spatialA_matmul_1 = torch.matmul(spatialA_transpose, spatialA_reshape_1)
        spatialA_Softmax = self.softmax(spatialA_matmul_1)
        spatialA_matmul_2 = torch.matmul(spatialA_reshape_1, spatialA_Softmax)
        spatialA_reshape_2 = torch.reshape(spatialA_matmul_2, [-1, self.band, 7, 7])
        spatial_feature = input_spatial + spatialA_reshape_2

        spatial_conv = self.spatial_conv(spatial_feature)
        spatial_bn = self.spatial_bn(spatial_conv)
        spatial_relu = self.relu(spatial_bn)
        return spatial_relu

    def spectral_attention(self, input_spectral):
        spectral_reshape_1 = torch.reshape(input_spectral, [-1, self.band, 7 * 7])
        spectral_transpose = spectral_reshape_1.permute(0, 2, 1)
        spectral_matmul_1 = torch.matmul(spectral_reshape_1, spectral_transpose)
        spectral_softmax =self.softmax(spectral_matmul_1)
        spectral_matmul_2 = torch.matmul(spectral_softmax, spectral_reshape_1)
        spectral_reshape_2 = spectral_matmul_2.view(-1, self.band, 7, 7)
        spectral_feature = input_spectral + spectral_reshape_2

        spectral_biases = self.spectral_biases(spectral_feature)
        spectral_bn = self.spectral_bn(spectral_biases)
        spectral_relu = self.relu(spectral_bn)
        return spectral_relu

    def feature_fusion(self, input_spatial, input_spectral):
        # 将输入张量重新调整形状以适应操作
        spatial_reshape = input_spatial.view(-1, self.band, 7 * 7)
        spectral_reshape = input_spectral.view(-1, self.band, 7 * 7)

        # 转置操作
        spatial_transpose = spatial_reshape.permute(0, 2, 1)
        spectral_transpose = spectral_reshape.permute(0, 2, 1)

        # 矩阵乘法
        spe_to_spa = torch.matmul(spectral_transpose, spatial_reshape)
        spa_to_spe = torch.matmul(spectral_reshape, spatial_transpose)

        # Softmax 操作
        spatial_soft = F.softmax(spe_to_spa, dim=-1)
        spectral_soft = F.softmax(spa_to_spe, dim=-1)

        # 矩阵乘法与相加
        spatial_final = torch.matmul(spectral_reshape, spatial_soft) + spatial_reshape
        spectral_final = torch.matmul(spectral_soft, spatial_reshape) + spectral_reshape

        # 最终特征相加
        feature_sum = spatial_final + spectral_final

        return spatial_final, spectral_final, feature_sum

    def logits_spatial(self, input_spatial):
        spatial_reshape = input_spatial.view(-1, self.band * 7 * 7)
        spatial_f1 = self.f1_spa(spatial_reshape)
        spatial_relu = self.relu(spatial_f1)
        spatial_drop = self.dropout_spa(spatial_relu)
        spatial_logits = self.f2_spa(spatial_drop)
        return spatial_logits

    def logits_spectral(self, input_spectral):
        spectral_reshape = input_spectral.view(-1, 7 * 7 * self.band)
        spectral_logit1 = self.f1_spe(spectral_reshape)
        spectral_fc1 = self.relu(spectral_logit1)
        spectral_drop =self.dropout_spe(spectral_fc1)
        spectral_logits = self.f2_spe(spectral_drop)
        return spectral_logits

    def classer(self, input_fusion):
        fusion_reshape = input_fusion.view(-1, 7 * 7 * self.band)
        fusion_logit1 = self.f1_classer(fusion_reshape)
        fusion_fc1 =self.relu(fusion_logit1)
        fusion_drop = self.dropout_classe(fusion_fc1)
        logits = self.f2_classer(fusion_drop)
        prediction =self.softmax(logits)
        return logits, prediction

    def network(self, input):
        feature_res = self.res_block(input)
        feature_spatial = self.spatial_attention(feature_res)
        feature_spectral = self.spectral_attention(feature_res)
        spa_sum, spe_sum, fusion = self.feature_fusion(feature_spatial, feature_spectral)
        logit_spatial = self.logits_spatial(spa_sum)
        logit_spectral = self.logits_spectral(spe_sum)
        logit, predict = self.classer(fusion)
        return logit_spatial, logit_spectral, logit, predict

    def forward(self, x):
        # return self.feature_fusion(self.spatial_attention(x),self.spectral_attention(x))[0]
        return self.network(x)

#
# import torch.optim as optim
#
# # 定义模型、损失函数和优化器
# model = HSINet()  # 初始化模型
# criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
#
# # 随机生成数据
# batch_size = 4
# depth, height, width = 103, 13, 13
# input_data = torch.randn(batch_size, 1, depth, height, width)  # 生成输入数据
# targets = torch.randint(0, 10, (batch_size,))  # 随机生成目标值（示例中假设有10个类别）
#
# # 进行训练
# num_epochs = 5
# for epoch in range(num_epochs):
#     model.train()  # 设置为训练模式
#     optimizer.zero_grad()  # 梯度清零
#
#     # 前向传播
#     outputs = model(input_data)
#
#     # 计算损失
#     loss = criterion(outputs, targets)
#
#     # 反向传播和优化
#     loss.backward()
#     optimizer.step()
#
#     # 打印当前 epoch 的损失
#     print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f}")
