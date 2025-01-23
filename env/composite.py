from env import core

Map = core.FunctionExpr.with_node(
    core.InnerArg(
        core.Loop(
            core.FunctionExpr.with_node(
                core.InstructionGroup(
                    core.Assign(
                        core.Integer(1),
                        core.NestedArg.from_raw(
                            node=core.Param.from_int(1),
                            indices=(1, 1),
                        ),
                    ),
                    core.Assign(
                        core.Integer(2),
                        core.NestedArg.from_raw(
                            node=core.Param.from_int(1),
                            indices=(1, 2),
                        )
                    ),
                    core.Assign(
                        core.Integer(3),
                        core.NestedArg.from_raw(
                            node=core.Param.from_int(1),
                            indices=(1, 3),
                        )
                    ),
                    core.Assign(
                        core.Integer(4),
                        core.Next(core.Var.from_int(1))
                    ),
                    core.Return.with_node(
                        core.If(
                            core.IsEmpty(core.Var.from_int(4)),
                            core.LoopGuard.with_args(
                                condition=core.IBoolean.false(),
                                result=core.Var.from_int(3),
                            ),
                            core.LoopGuard.with_args(
                                condition=core.IBoolean.true(),
                                result=core.DefaultGroup(
                                    core.NestedArg.from_raw(
                                        node=core.Var.from_int(4),
                                        indices=(1, 1),
                                    ),
                                    core.Var.from_int(2),
                                    core.Add(
                                        core.Var.from_int(3),
                                        core.DefaultGroup(
                                            core.FunctionCall(
                                                core.Var.from_int(2),
                                                core.DefaultGroup(
                                                    core.NestedArg.from_raw(
                                                        node=core.Var.from_int(4),
                                                        indices=(1, 2),
                                                    )
                                                ),
                                            )
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            core.Optional(
                core.DefaultGroup(
                    core.GroupIterator.with_node(
                        core.Param.from_int(1),
                    ),
                    core.Param.from_int(2),
                    core.DefaultGroup(),
                )
            ),
        ),
        core.NodeArgIndex(1),
    ),
)
