from env import core

Map = core.FunctionExpr.with_node(
    core.InstructionGroup(
        core.Loop(
            core.FunctionExpr.with_node(
                core.InstructionGroup(
                    core.Assign(
                        core.Integer(1),
                        core.InnerArg(core.Param.from_int(1), core.Integer(1))
                    ),
                    core.Assign(
                        core.Integer(2),
                        core.InnerArg(core.Param.from_int(1), core.Integer(2))
                    ),
                    core.Assign(
                        core.Integer(3),
                        core.Next(core.Var.from_int(1))
                    ),
                    core.If(
                        core.IsEmpty(core.Var.from_int(3)),
                        core.LoopGuard.with_args(
                            condition=core.IBoolean.false(),
                            result=core.Param.from_int(3),
                        ),
                        core.LoopGuard.with_args(
                            condition=core.IBoolean.true(),
                            result=core.Add(
                                core.Param.from_int(3),
                                core.DefaultGroup(
                                    core.FunctionCall(
                                        core.Var.from_int(2),
                                        core.DefaultGroup(
                                            core.InnerArg(core.Var.from_int(3), core.Integer(1))
                                        ),
                                    )
                                ),
                            )
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
            )
        ),
    ),
)