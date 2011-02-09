"""Test suite for rain.py

Run the test suite with::

  $ nosetests

or::

  $ nosetests test_rain.py
"""
from __future__ import division
import numpy as np
from nose.tools import assert_equal, with_setup
# Module under test
import rain


def set_up():
    """Test environment set-up.
    """
    global n_grid, n_time, g, H, dt, dx, ho, gu, gh, u, h
    n_grid = 9
    n_time = 5
    g = 980                     # acceleration due to gravity [cm/s^2]
    H = 1                       # water depth [cm]
    dt = 0.001                  # time step [s]
    dx = 1                      # grid spacing [cm]
    ho = 0.01                   # initial perturbation of surface [cm]
    gu = g * dt / dx
    gh = H * dt / dx
    u = rain.Quantity(n_grid, n_time)
    h = rain.Quantity(n_grid, n_time)


@with_setup(set_up)
def test_Quantity_init():
    """Instance of Quantity class has expected attributes.
    """
    qty = rain.Quantity(n_grid, n_time)
    assert_equal(qty.n_grid, n_grid)
    assert_equal(qty.prev.shape, (n_grid, ))
    assert_equal(qty.now.shape, (n_grid, ))
    assert_equal(qty.next.shape, (n_grid, ))
    assert_equal(qty.store.shape, (n_grid, n_time))


@with_setup(set_up)
def test_Quantity_store_timestep():
    """store_timestep() method copies time step values to storage array
    """
    rain.initial_conditions(u, h, ho)
    h.store_timestep(0, 'prev')
    np.testing.assert_equal(h.store[:, 0], h.prev)


@with_setup(set_up)
def test_Quantity_shift():
    """shift() method copies .now to .prev, and .next to .now
    """
    u.now = np.ones(n_grid)
    u.next = np.ones(n_grid) * 2
    u.shift()
    np.testing.assert_equal(u.prev, np.ones(n_grid))
    np.testing.assert_equal(u.now, np.ones(n_grid) * 2)
    h.now = np.ones(n_grid) * 3
    h.next = np.ones(n_grid) * 4



@with_setup(set_up)
def test_initial_conditions():
    """initial_conditions() sets u.prev & h.prev correctly
    """
    rain.initial_conditions(u, h, ho)
    # u.prev values
    np.testing.assert_equal(u.prev, np.zeros(n_grid))
    # h.prev values
    midpoint = n_grid // 2
    np.testing.assert_equal(h.prev[:midpoint], np.zeros(midpoint))
    assert_equal(h.prev[midpoint], ho)
    np.testing.assert_equal(h.prev[midpoint + 1:], np.zeros(midpoint))


@with_setup(set_up)
def test_first_time_step():
    """first_time_step() sets u.now and h.now correctly
    """
    rain.first_time_step(u, h, g, H, dt, dx, ho, gu, gh, n_grid)
    midpoint = n_grid // 2
    # u.now values
    np.testing.assert_equal(u.now[1:midpoint - 1], np.zeros(midpoint - 2))
    assert_equal(u.now[midpoint - 1], -gu * ho / 2)
    assert_equal(u.now[midpoint], 0)
    assert_equal(u.now[midpoint + 1], gu * ho / 2)
    np.testing.assert_equal(u.now[midpoint + 2:n_grid - 1],
                            np.zeros(midpoint - 2))
    # h.now values
    np.testing.assert_equal(h.now[1:midpoint], np.zeros(midpoint - 1))
    assert_equal(h.now[midpoint], ho - g * H * ho * dt ** 2 / (4 * dx ** 2))
    np.testing.assert_equal(h.now[midpoint + 1:n_grid - 1], 
                            np.zeros(midpoint - 1))


@with_setup(set_up)
def test_boundary_conditions():
    """boundary_conditions() sets boundary condition values correctly
    """
    h.now[1] = h.now[n_grid - 2] = 3.1415927
    rain.boundary_conditions(u.now, h.now, n_grid)
    assert_equal(u.now[0], 0)
    assert_equal(u.now[n_grid - 1], 0)
    assert_equal(h.now[0], 3.1415927)
    assert_equal(h.now[n_grid - 1], 3.1415927)
