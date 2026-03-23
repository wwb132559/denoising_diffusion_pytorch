class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

class Node_d:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None

    def append(self, data):
        new_node = Node(data)
        if self.is_empty():
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def prepend(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def insert_after_node(self, prev_node_data, data):
        new_node = Node(data)
        current = self.head
        while current.data != prev_node_data:
            current = current.next
            if current is None:
                print("The previous node data is not in the list.")
                return
        new_node.next = current.next
        current.next = new_node

    def delete_node(self, key):
        current = self.head
        if current.data == key:
            self.head = current.next
            current = None
            return
        prev = None
        while current.data != key:
            prev = current
            current = current.next
            if current is None:
                return
        prev.next = current.next
        current = None

    def reverse(self):
        S = Solution()
        p = S.reverseList(self.head)
        self.head = p

    def print_list(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

class Solution:
    def printListFromTailTOHead(self, LinkedList):
        if LinkedList is None:
            return []
        result = []
        while LinkedList:
            result.append(LinkedList.data)
            LinkedList = LinkedList.next
        result.reverse()
        return result

    def reverseList(self, head):
        current = head
        prev = None
        while current:
            next = current.next
            current.next = prev
            prev = current
            current = next
        return prev




class DoubleLinkedList:
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None

    def append(self, data):
        new_node = Node_d(data)
        if self.is_empty():
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        new_node.prev = current

    def prepend(self, data):
        new_node = Node_d(data)
        if self.is_empty():
            self.head = new_node
            return
        new_node.next = self.head
        self.head.prev = new_node
        self.head = new_node

    def insert_after_node(self, prev_node_data, data):
        new_node = Node_d(data)
        current = self.head
        prev = None
        while current.data != prev_node_data:
            current = current.next
            if current is None:
                print("The previous node data is not in the list.")
                return
        new_node.next = current.next
        new_node.prev = current
        if current.next:
            current.next.prev = new_node
        current.next = new_node

    def delete_node(self, key):
        current = self.head
        while current.data != key:
            current = current.next
            if current is None:
                return
        if current.prev:
            current.prev.next = current.next
        else:
            self.head = current.next
            self.head.prev = None
        if current.next:
            current.next.prev = current.prev
        current = None

    def print_list(self):
        current = self.head
        while current:
            print(current.data, end=" <-> ")
            current = current.next
        print("None")


l1 = LinkedList()
l1.append(1)
l1.append(3)
l1.append(4)
l1.print_list()

l1.prepend(-1)
l1.print_list()

l1.insert_after_node(1, 1.5)
l1.print_list()  # 输出: 0 -> 1 -> 1.5 -> 2 -> 3 -> None

l1.delete_node(1.5)
l1.print_list()  # 输出: 0 -> 1 -> 2 -> 3 -> None

l1.reverse()
l1.print_list()

# S = Solution()
# S.reverseList(l1.head)
# l1.print_list()