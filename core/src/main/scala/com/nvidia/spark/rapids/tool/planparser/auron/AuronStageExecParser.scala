/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.tool.planparser.auron

import com.nvidia.spark.rapids.tool.planparser.WholeStageExecParserBase
import com.nvidia.spark.rapids.tool.qualification.PluginTypeChecker

import org.apache.spark.sql.rapids.tool.AppBase
import org.apache.spark.sql.rapids.tool.util.plangraph.AuronSparkPlanGraphCluster

/**
 * Parses Auron Stage Exec equivalent to Spark's `WholeStageCodegenExec`.
 * This can be extended if specific parsing is needed for a particular Stage operator.
 *
 * @see Resource file in `auronOperatorMappings` for the mapping of Auron operators
 *      to Spark operators
 */
case class AuronStageExecParser(
    node: AuronSparkPlanGraphCluster,
    checker: PluginTypeChecker,
    sqlID: Long,
    app: AppBase,
    reusedNodeIds: Set[Long],
    nodeIdToStagesFunc: Long => Set[Int])
  extends WholeStageExecParserBase(node, checker, sqlID, app, reusedNodeIds, nodeIdToStagesFunc)
