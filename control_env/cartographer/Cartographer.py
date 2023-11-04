import matplotlib.pyplot as plt
from control_env.core.System import System


class Graph:
    def __init__(self, name, num_sub_graph=1):
        fig, axs = plt.subplots(num_sub_graph, 1, figsize=[12.8, 8])
        self.fig = fig
        self.axs = axs
        self.fig.suptitle(name)

    def plot_system_data(self, system: System, name=None):

        assert len(system.state_names) <= len(self.axs)
        t, x = system.history()
        while len(system.state_names) < x.shape[1]:
            system.state_names.append('')
        for i in range(x.shape[1]):
            self.axs[i].plot(t, x[:, i:i + 1], label=system.name if name is None else name)
            self.axs[i].set_xlabel('Time')
            self.axs[i].set_ylabel(system.state_names[i])
            self.axs[i].legend(bbox_to_anchor=(1.2, 1.0))
        self.fig.tight_layout()
        self.fig.align_labels()

    def plot_single_data(self, system: System, name, num_graph):
        t, x = system.history()
        index = system.state_names.index(name)
        self.axs[num_graph].plot(t, x[:, index:index + 1], label=name)
        self.axs[num_graph].set_xlabel('Time')
        self.axs[num_graph].set_ylabel(name)
        self.axs[num_graph].legend(bbox_to_anchor=(1.2, 1.0))
        self.fig.tight_layout()
        self.fig.align_labels()

    def show(self):
        self.fig.show()
