from  pyplanscoring.constraints.constraints import MayoConstraint


def test_MayoConstraint():
    """
        Test class MayoConstraint
    """

    constrain = 'D95%[cGy] > 7000'
    ctr = MayoConstraint()
    ctr.read(constrain)

    assert ctr.constraint_value == 7000.0
    assert ctr.discriminator == 2


ctr = MayoConstraint()


def test_query():
    constrain = 'D95%[cGy] > 7000'
    ctr.read(constrain)
    assert ctr.query.to_string() == 'D95%[cGy]'


def test_discriminator():
    constrain = 'D95%[cGy] < 7000'
    ctr.read(constrain)
    assert ctr.discriminator == 0

    constrain = 'D95%[cGy] <= 7000'
    ctr.read(constrain)
    assert ctr.discriminator == 1

    constrain = 'D95%[cGy] > 7000'
    ctr.read(constrain)
    assert ctr.discriminator == 2

    constrain = 'D95%[cGy] >= 7000'
    ctr.read(constrain)
    assert ctr.discriminator == 3

    constrain = 'D95%[cGy] = 7000'
    ctr.read(constrain)
    assert ctr.discriminator == 4


def test_constraint_value():
    constrain = 'D95%[cGy] >= 7000'
    ctr.read(constrain)
    assert ctr.constraint_value == 7000

    constrain = 'D95%[Gy] >= 47.5'
    ctr.read(constrain)
    assert ctr.constraint_value == 47.5


def test_read():
    constrain = 'D95%[cGy] > 7000'
    ctr.read(constrain)
    assert ctr.query.to_string() == 'D95%[cGy]'
    assert ctr.discriminator == 2
    assert ctr.constraint_value == 7000


def test_write():
    constrain = 'D95%[cGy] < 7000'
    ctr.read(constrain)
    assert ctr.write() == constrain

    constrain = 'D95%[cGy] <= 7000'
    ctr.read(constrain)
    assert ctr.write() == constrain

    constrain = 'D95%[cGy] > 7000'
    ctr.read(constrain)
    assert ctr.write() == constrain

    constrain = 'D95%[cGy] >= 7000'
    ctr.read(constrain)
    assert ctr.write() == constrain

    constrain = 'D95%[cGy] = 7000'
    ctr.read(constrain)
    assert ctr.write() == constrain
