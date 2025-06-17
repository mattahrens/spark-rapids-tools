/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

import com.nvidia.spark.rapids.tool.ApacheSparkEventLog
import com.nvidia.spark.rapids.tool.planparser.gluten.GlutenPlanParser
import com.nvidia.spark.rapids.tool.qualification.PluginTypeChecker
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.scalatest.FunSuite

import org.apache.spark.sql.rapids.tool.profiling.ApplicationInfo
import org.apache.spark.sql.rapids.tool.util.plangraph.GlutenSparkPlanGraphNode


class GlutenPlanParserSuite extends FunSuite {
  // Create a mock ApplicationInfo that doesn't require an event log
  class MockApplicationInfo extends ApplicationInfo(
      new Configuration(),
      new ApacheSparkEventLog(new Path("dummy"))) {
    override def processEvents(): Unit = {} // Override to do nothing
  }

  test("GlutenBroadcastNestedLoopJoin is parsed as Spark BroadcastNestedLoopJoin") {
    val node = new GlutenSparkPlanGraphNode(
      1L,
      "GlutenBroadcastNestedLoopJoinExec",
      "GlutenBroadcastNestedLoopJoinExec",
      "BroadcastNestedLoopJoinExec",
      "BroadcastNestedLoopJoinExec",
      Seq.empty
    )
    val app = new MockApplicationInfo()
    val checker = new PluginTypeChecker()
    val execInfo = GlutenPlanParser.parseNode(node, 1L, checker, app)
    assert(execInfo.exec === "BroadcastNestedLoopJoinExec")
  }

  test("GlutenProject is parsed as Spark Project") {
    val node = new GlutenSparkPlanGraphNode(
      1L,
      "GlutenProjectExec",
      "GlutenProjectExec",
      "ProjectExec",
      "ProjectExec",
      Seq.empty
    )
    val app = new MockApplicationInfo()
    val checker = new PluginTypeChecker()
    val execInfo = GlutenPlanParser.parseNode(node, 1L, checker, app)
    assert(execInfo.exec === "ProjectExec")
  }

  test("GlutenShuffleMapStage is parsed as Spark WholeStageCodegen") {
    val node = new GlutenSparkPlanGraphNode(
      1L,
      "GlutenShuffleMapStage",
      "GlutenShuffleMapStage",
      "WholeStageCodegen",
      "WholeStageCodegen",
      Seq.empty
    )
    val app = new MockApplicationInfo()
    val checker = new PluginTypeChecker()
    val execInfo = GlutenPlanParser.parseNode(node, 1L, checker, app)
    assert(execInfo.exec === "WholeStageCodegen")
  }
}
