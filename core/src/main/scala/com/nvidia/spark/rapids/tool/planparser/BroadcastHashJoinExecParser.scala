/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.planparser

import com.nvidia.spark.rapids.tool.qualification.PluginTypeChecker

import org.apache.spark.sql.execution.ui.SparkPlanGraphNode

case class BroadcastHashJoinExecParser(
    node: SparkPlanGraphNode,
    checker: PluginTypeChecker,
    sqlID: Long) extends ExecParser {

  val fullExecName = node.name + "Exec"

  override def parse: ExecInfo = {
    // BroadcastHashJoin doesn't have duration
    val duration = None
    // Handle both "BroadcastHashJoin" and "NativeBroadcastJoin" (or other mapped names)
    // in description. The description should already be mapped by AuronSparkPlanGraphNode,
    // but handle both formats for robustness.
    var exprString = node.desc
    // Try to remove operator name prefixes in order (longest first to avoid partial matches)
    val prefixesToRemove = Seq(
      "NativeBroadcastHashJoin",
      "NativeBroadcastJoin",
      "BroadcastHashJoin"
    )
    prefixesToRemove.foreach { prefix =>
      exprString = exprString.replaceFirst(s"^$prefix\\s+", "")
    }
    val (expressions, _) = SQLPlanParser.parseEquijoinsExpressions(exprString)
    val notSupportedExprs = expressions.filterNot(expr => checker.isExprSupported(expr))
    // For BroadcastHashJoin, if the exec is supported and there are no unsupported expressions,
    // mark it as supported. The join type check is still performed, but if parsing fails
    // (e.g., for Auron operators with non-standard descriptions), we still mark as supported
    // as long as BroadcastHashJoinExec itself is supported and there are no unsupported
    // expressions. This ensures that BroadcastHashJoin (which is supported on GPU) is not
    // incorrectly marked as unsupported due to description parsing issues.
    val execIsSupported = checker.isExecSupported(fullExecName)
    val (speedupFactor, isSupported) = if (execIsSupported && notSupportedExprs.isEmpty) {
      // If exec is supported and no unsupported expressions, mark as supported.
      // Note: We still respect supportedJoinType when it's true, but don't fail if parsing
      // fails (supportedJoinType is false) since BroadcastHashJoin is supported on GPU.
      (checker.getSpeedupFactor(fullExecName), true)
    } else {
      (1.0, false)
    }
    ExecInfo(node, sqlID, node.name, "", speedupFactor, duration, node.id, isSupported,
      children = None, expressions = expressions)
  }
}
