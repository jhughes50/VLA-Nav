"""
    CIS 6200 -- Deep Learning Final Project
    functions to log the loss
    May 2024
"""

class LossLogger:

    def __init__(self, output_dir, model):

        self.output_dir_ = output_dir
        self.model_ = model

        self.file_ = open(output_dir+"loss-%s.log" %model, 'w')

    def __del__(self):
        self.file_.close()

    def log(self, loss, idx=None):
        loss = loss.item()
        if idx == None:
            self.file_.write(str(loss)+"\n")
        else:
            self.file_.write("%s--%s\n" %(idx,loss))
