import numpy as np
import matplotlib.pyplot as plt

#顯示學習曲線
def get_train_state(train_path, losses, eType, epochs=20):
    x = np.arange(1, len(losses)+1)
    y = losses
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')  
    plt.xticks(x)
    plt.plot(x, y)
    plt.savefig(train_path+'/{}_E_{}.jpg'.format(eType, epochs))
    plt.show()
