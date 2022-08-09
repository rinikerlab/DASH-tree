import numpy as np


class develop_node:
    def __init__(self, atom=None, level=None, data=None, attention_data=None, parent_attention=0):
        self.children = []
        self.level = level
        self.parent_attention = parent_attention
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        if isinstance(attention_data, np.ndarray):
            self.attention_data = attention_data
        else:
            self.attention_data = np.array(attention_data)
        self.average = 0
        self.atom = atom

    def __repr__(self):
        if self.level == 0:
            return f"node --- lvl: {self.level}, Num=1"
        else:
            return f"node --- lvl: {self.level}, Num={str(self.data.size)}, Mean={float(self.average):.4f}, std={np.std(self.data):.4f}, fp={self.atom}"

    def __eq__(self, other):
        return self.level == other.level and self.atom == other.atom

    def add_child(self, child):
        self.children.append(child)

    def add_node(self, branch: list):
        current_parent = self
        for layer, node in enumerate(branch):
            if node in current_parent.children:
                correct_child = current_parent.children[current_parent.children.index(node)]
                correct_child.data = np.append(correct_child.data, node.data)
                correct_child.attention_data = np.append(correct_child.attention_data, node.attention_data)
                current_parent = correct_child
            else:
                current_parent.add_child(node)
                current_parent = current_parent.children[-1]

    def max_attention(self):
        return np.max(self.attention_data)

    def max_attention_index(self):
        return np.argmax(self.attention_data)

    def update_average(self):
        if self.level != 0:
            if self.data.size == 1:
                self.average = self.data
            elif self.data.size > 1:
                self.average = np.mean(self.data)
            else:
                self.average = np.NAN
        for child in self.children:
            child.update_average()

    def get_node_result_and_std_and_attention_and_length(self, attention_percentage: int = 0.0):
        """
        Returns averaged properties of a node
        Returns the result of the node and the standard deviation of the node and the attention of the node

        Parameters
        ----------
        attention_percentage : int, optional
            threshold for averaging, by default 0.95

        Returns
        -------
        list[float, float, float, int]
            [result, stdDeviation, attention, length]
        """
        if self.level == 0:
            return (np.float32(np.nan), np.float32(np.nan), np.float32(np.nan), 0)
        if self.data.size == 1:
            return (np.float32(self.data), np.float32(np.nan), np.float32(self.attention_data), 1)
        if self.data.size > 1:
            selected_data = self.data[(self.attention_data + self.parent_attention) > attention_percentage]
            if selected_data.size == 0:
                return (np.mean(self.data), np.std(self.data), np.mean(self.attention_data), self.data.size)
            else:
                return (np.mean(selected_data), np.std(selected_data), np.mean(self.attention_data), selected_data.size)
