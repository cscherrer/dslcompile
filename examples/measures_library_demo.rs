let joint_result = ctx.eval(
    &joint_log_density,
    hlist![0.0, 1.0, 1.0, 2.0, 0.5, 3.0],
); 