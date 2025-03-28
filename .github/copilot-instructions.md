# Project Structure Guide

This document outlines the main components and architecture of the AutoMath project, which implements a mathematical agent system.

## Environment (`env/`)

The `env/` directory contains the core environment components that power the mathematical agent system:

- `action.py` - Defines the abstract base `Action` class that serves as the foundation for all concrete actions
- `action_impl.py` - Implements various concrete action classes (e.g., `ActionMeta`, `StateMeta`, `ManageScratch`, `DefineScratch`, `UpdateScratch`, `ManageArgsGroup`)
- `base_agent.py` - Contains the base agent class implementation that other agents inherit from
- `composite.py` - Facilitates composition of multiple components into more complex structures
- `core.py` - Provides core functionality and base classes for the fundamental nodes in the system
- `env_utils.py` - Offers utility functions to support the environment's operations
- `environment.py` - Implements the main environment that agents interact with
- `full_state.py` - Represents the complete state including meta information, current main state, and history of past states and actions
- `goal_env.py` - Extends the environment to be goal-oriented (terminates when specific goals are achieved)
- `meta_env.py` - Implements a meta-environment for higher-level control and reasoning
- `node_data.py` - Defines the node data structure used by agents (flattens tree-structured nodes into a matrix format)
- `node_types.py` - Implements specialized node types that extend the core nodes for specific purposes
- `reward.py` - Provides mechanisms for calculating and distributing rewards in the reinforcement learning framework
- `state.py` - Handles the core state representation and management functions
- `symbol.py` - Manages symbol handling and manipulation for mathematical expressions and printing purposes
- `trainer.py` - Contains components for training agents through various learning approaches

## Test Suite (`test_suite/`)

The `test_suite/` directory contains comprehensive tests that verify the system's functionality:

- `arithmetic_test.py` - Tests mathematical arithmetic operations and their implementations
- `basic_test.py` - Verifies fundamental functionality of the system's components
- `boolean_test.py` - Tests boolean operations and logical reasoning capabilities
- `control_flow_test.py` - Validates control flow mechanisms and decision-making processes
- `indices_test.py` - Tests operations related to indexing and array manipulations
- `test_root.py` - Serves as the root test configuration and main entry point for all tests
- `test_utils.py` - Provides utility functions and helpers to support testing activities

### Action Implementation Tests (`test_suite/action_impl/`)

This subdirectory contains specialized tests for the various action implementations:

- `action_00_action_meta.py` - Tests the functionality of action metadata handling
- `action_01_state_meta.py` - Validates state metadata manipulation operations
- `action_02_manage_scratch.py` - Tests scratch space management capabilities
- `action_03_define_scratch.py` - Verifies scratch space definition functionality
- `action_04_update_scratch.py` - Tests mechanisms for updating scratch workspaces
- `action_05_manage_args_group.py` - Validates argument group management operations
