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

// 3rd party imports
import org.scalatest.FunSuite

class GlutenParseHelperSuite extends FunSuite {

  test("isGlutenApp detects Gluten app correctly") {
    val glutenProps = Map(
      "spark.gluten.enabled" -> "true",
      "spark.gluten.sql.columnar.backend.ch" -> "clickhouse"
    )
    assert(GlutenParseHelper.isGlutenApp(glutenProps))

    val nonGlutenProps = Map(
      "spark.gluten.enabled" -> "false",
      "spark.gluten.sql.columnar.backend.ch" -> "clickhouse"
    )
    assert(!GlutenParseHelper.isGlutenApp(nonGlutenProps))
  }

  test("isGlutenNode detects Gluten nodes correctly") {
    assert(GlutenParseHelper.isGlutenNode("GlutenBroadcastHashJoinExec"))
    assert(GlutenParseHelper.isGlutenNode("GlutenFilterExec"))
    assert(!GlutenParseHelper.isGlutenNode("BroadcastHashJoinExec"))
    assert(!GlutenParseHelper.isGlutenNode("FilterExec"))
  }

  test("mapGlutenToSpark maps Gluten operators to Spark operators") {
    val testCases = Seq(
      "GlutenBroadcastHashJoinExec" -> "BroadcastHashJoinExec",
      "GlutenFilterExec" -> "FilterExec",
      "GlutenProjectExec" -> "ProjectExec",
      "GlutenHashAggregateExec" -> "HashAggregateExec",
      "GlutenSortExec" -> "SortExec",
      "GlutenColumnarToRowExec" -> "ColumnarToRowExec"
    )

    testCases.foreach { case (glutenName, expectedSparkName) =>
      assert(GlutenParseHelper.mapGlutenToSpark(glutenName) === expectedSparkName)
    }

    // Test unknown Gluten operator
    val unknownOp = "GlutenUnknownExec"
    assert(GlutenParseHelper.mapGlutenToSpark(unknownOp) === unknownOp)
  }
}
