class TVAModel_(nn.Module): # trimodal self-attn model  
    def __init__(self, params):
        super(TVAModel_, self).__init__()
        rnn = nn.LSTM if params.rnntype == "lstm" else nn.GRU
        self.text_encoder = rnn(input_size=params.txt_dim, hidden_size=params.txt_rnnsize,
                                num_layers=params.txt_rnnnum, dropout=params.txt_rnndp, bidirectional=params.rnndir,
                                batch_first=True)
        self.video_conv = nn.Conv1d(in_channels=32, out_channels=25, kernel_size=1) # 825 params
        self.video_encoder = rnn(input_size=params.vid_dim, hidden_size=params.vid_rnnsize,
                                 num_layers=params.vid_rnnnum, dropout=params.vid_rnndp, bidirectional=params.rnndir,
                                 batch_first=True)
        self.audio_conv = nn.Conv1d(in_channels=1000, out_channels=500, kernel_size=1)
        self.audio_encoder = rnn(input_size=params.aud_dim, hidden_size=params.aud_rnnsize,
                                 num_layers=params.aud_rnnnum, dropout=params.aud_rnndp, bidirectional=params.rnndir,
                                 batch_first=True)
        if params.rnndir:
            self.mha_t = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.txt_nh,
                                                 dropout=params.txt_dp, batch_first=True)
            self.mha_v = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.vid_nh,
                                               dropout=params.vid_dp, batch_first=True)
            self.mha_a = nn.MultiheadAttention(embed_dim=2 * params.rnnsize, num_heads=params.aud_nh,
                                               dropout=params.aud_dp, batch_first=True)
            self.concat_linear = nn.Linear(in_features=2 * 2 * params.rnnsize, out_features= params.rnnsize)
            self.classifier = nn.Linear(in_features= params.rnnsize, out_features=params.output_dim)

    def forward(self, x_txt, x_vid, x_mfcc, emb_dp=0.25):
        # text branch
        x_txt = F.dropout(x_txt, p=emb_dp, training=self.training) # [32, 128, 300]
        x_txt, h = self.text_encoder(x_txt) # [32, 128, 400]
        x_txt, _ = self.mha_t(x_txt,x_txt,x_txt) # [32, 128, 400]
        x_txt2 = torch.mean(x_txt, dim=1)
        # video branch
        x_vid = self.video_conv(x_vid)
        x_vid, h = self.video_encoder(x_vid)
        x_vid, _ = self.mha_v(x_vid, x_vid, x_vid) # [32, 25, 400]
        x_vid2 = torch.mean(x_vid, dim=1)
        # audio branch
        x_mfcc = self.audio_conv(x_mfcc)
        x_mfcc, h = self.audio_encoder(x_mfcc)
        x_mfcc, _ = self.mha_a(x_mfcc, x_mfcc, x_mfcc)
        x_mfcc2 = torch.mean(x_mfcc, dim=1)
        x_tva1 = torch.stack((x_txt2, x_vid2, x_mfcc2), dim=1) # [32, 3, 240]
        x_tva1_mean, x_tva1_std = torch.std_mean(x_tva1, dim=1)
        x_tva = torch.cat((x_tva1_mean, x_tva1_std), dim=1) # [32, 480]
        x_tva = self.concat_linear(x_tva)
        y = self.classifier(x_tva) # [32, 7]
        return y, x_tva
