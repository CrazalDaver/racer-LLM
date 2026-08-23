"""
United simulation entry point for the two IBR policy families.
Used in main simulation loop to coordinate parameter and trajectory IBR updates before the get_action MPC calls.
Jointly updating both cars' predictions.
"""


def plan_ibr_joint_actions(policies, states, car_s, car_e, sim_time=None):
    """Run parameter updates, then cache direct trajectory-IBR actions."""
    # Local imports avoid coupling this facade to concrete implementations
    # while the policy.ibr package is being initialized.
    from .mpc_parameter_ibr import plan_parameter_ibr_policy_params
    from .trajectory_ibr import plan_trajectory_ibr_actions

    # Parameter IBR updates theta here. MPCPolicy.get_action later converts
    # the selected parameters into a low-level control.
    plan_parameter_ibr_policy_params(
        policies, states, car_s, car_e, sim_time=sim_time
    )
    # policy.params
    # policy.predicted_frenet_plan

    # Trajectory IBR directly optimizes and caches its next low-level action.
    plan_trajectory_ibr_actions(
        policies, states, car_s, car_e, sim_time=sim_time
    )
