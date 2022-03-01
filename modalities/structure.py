from typing import Optional

# TODO: you should be able to work with tensors where the grid is on axes 1 and 3 and the embedding is on axis 2, etc.


class Structure:
    pass


class Flat(Structure):
    """
    Represents data where
    - each element's index location has a specific, fixed meaning

    For example: vector embeddings
    """
    depth: Optional[int]


class Set(Structure):
    """
    Represents data where
    - the tensor value index should not be used

    For example, a bag of words
    """
    depth: Optional[int]


class Grid(Structure):
    """
    Represents data where
    - there is a concept of location associated with each element
    - but no directionality (distance is symmetric)

    For example, an image grid
    """
    depth: Optional[int]


class Seq(Structure):
    """
    Represents data where
    - there is a concept of location associated with each element
    - and definite directionality (distance is not symmetric)

    For example, a sequence of tokens, an action trajectory
    """
    pass


class Onehot(Flat):
    """
    Represents data where
    - each element's index location has a specific, fixed meaning
    - only one element can be active at a time
    """

    @property
    def categories(self):
        return self.depth
