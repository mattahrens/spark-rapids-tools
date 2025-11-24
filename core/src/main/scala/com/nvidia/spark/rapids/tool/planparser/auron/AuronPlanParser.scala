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

import com.nvidia.spark.rapids.tool.planparser._
import com.nvidia.spark.rapids.tool.qualification.PluginTypeChecker

import org.apache.spark.sql.rapids.tool.AppBase
import org.apache.spark.sql.rapids.tool.util.plangraph.AuronSparkPlanGraphNode

object AuronPlanParser {
  /**
   * Parses an Auron node, using a specific Auron parser if available.
   * If no Auron specific parser is defined, use Spark CPU equivalent
   * ExecParser.
   *
   * @return Parsed ExecInfo
   */
  def parseNode(
      node: AuronSparkPlanGraphNode,
      sqlID: Long,
      checker: PluginTypeChecker,
      app: AppBase): ExecInfo = {
    // Currently, Auron nodes can fall back to Spark CPU parser
    // If specific Auron parsers are needed in the future, add them here
    // similar to how PhotonBroadcastNestedLoopJoinExecParser is handled
    SQLPlanParser.parseSparkNode(node, sqlID, checker, app)
  }
}
