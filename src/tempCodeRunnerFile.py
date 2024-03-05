if self.layer_id == 0 and args.pre_ffn > 0:
            #     x = x + self.lif1(self.ffnPre(self.ln1(x)).permute(1, 0, 2)).permute(1, 0, 2)
            # else:
            #     x = x + self.lif1(self.att(self.ln1(x)).permute(1, 0, 2)).permute(1, 0, 2)
