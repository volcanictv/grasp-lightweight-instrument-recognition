import re, shutil, os
p = os.path.expanduser("~/baselines/GraSP/TAPIS_gtbox/tapis/models/head_helper.py")
if not os.path.exists(p + ".orig"): shutil.copy(p, p + ".orig")
s = open(p).read()
a = "        self.projection = nn.Linear(1024, num_classes, bias=True)\n        self.act_func = act_func\n        # Softmax for evaluation and testing."
assert s.count(a) == 1
s = s.replace(a, "        # pooled video features (1024) concatenated with the projected detector features (1024)\n        self.projection = nn.Linear(2048, num_classes, bias=True)\n        self.act_func = act_func\n        # Softmax for evaluation and testing.")
old_fwd = s[s.index("    def forward(self, inputs, bboxes=None, features=None, **kwargs):\n        assert (\n            len(inputs) == self.num_pathways"):s.index("class ResNetBasicHead(nn.Module):")]
new_fwd = '''    def forward(self, inputs, bboxes=None, features=None, boxes_mask=None, **kwargs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        # TAPIS loader format: boxes (B, M, 4) padded, boxes_mask (B, M); the region list of the AVA-style head is (K, 5) = batch index and box of the valid boxes
        mask = boxes_mask.bool()
        batch_idx = torch.arange(mask.shape[0], device=mask.device, dtype=bboxes.dtype).view(-1, 1).expand_as(mask)[mask]
        rois = torch.cat([batch_idx[:, None], bboxes[mask]], dim=1)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, rois)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # K C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection_pathways(x)

        features = self.projection_faster(features[mask])
        x = torch.cat((x, features), axis=1)

        x = self.projection(x)

        if self.act_func == "sigmoid" or not self.training:
            x = self.act(x)

        return x

'''
s = s.replace(old_fwd, new_fwd)
open(p, "w").write(s); print("patched; backup at", p + ".orig")
