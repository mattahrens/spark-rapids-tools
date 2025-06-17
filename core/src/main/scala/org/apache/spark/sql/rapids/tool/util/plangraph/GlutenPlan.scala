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

package org.apache.spark.sql.rapids.tool.util.plangraph

import scala.collection.mutable

import com.nvidia.spark.rapids.tool.planparser.GlutenParseHelper

import org.apache.spark.sql.execution.ui.{SparkPlanGraphCluster, SparkPlanGraphNode, SQLPlanMetric}

/**
 * Extension of SparkPlanGraphNode to handle Gluten nodes.
 * Note:
 * - glutenName and glutenDesc are the name and description of the Gluten node
 * - sparkName and sparkDesc are the name and description of the equivalent Spark node
 */
class GlutenSparkPlanGraphNode(
    id: Long,
    val glutenName: String,
    val glutenDesc: String,
    sparkName: String,
    sparkDesc: String,
    metrics: collection.Seq[SQLPlanMetric])
  extends SparkPlanGraphNode(id, sparkName, sparkDesc, metrics)

object GlutenSparkPlanGraphNode {
  def from(node: SparkPlanGraphNode): GlutenSparkPlanGraphNode = {
    val sparkName = if (node.name.contains("Unknown")) {
      "Unknown operation"
    } else {
      GlutenParseHelper.mapGlutenToSpark(node.name)
    }
    val sparkDesc = if (node.desc.contains("Unknown")) {
      "Unknown operation"
    } else {
      GlutenParseHelper.mapGlutenToSpark(node.desc)
    }
    new GlutenSparkPlanGraphNode(node.id, node.name, node.desc, sparkName, sparkDesc, node.metrics)
  }
}

/**
 * Extension of SparkPlanGraphCluster to handle Gluten nodes that are
 * mapped to WholeStageCodegen.
 * Note:
 * - glutenName and glutenDesc are the name and description of the Gluten node
 * - name and desc are the name and description of the equivalent Spark node
 */
class GlutenSparkPlanGraphCluster(
    id: Long,
    val glutenName: String,
    val glutenDesc: String,
    sparkName: String,
    sparkDesc: String,
    nodes: mutable.ArrayBuffer[SparkPlanGraphNode],
    metrics: collection.Seq[SQLPlanMetric])
  extends SparkPlanGraphCluster(id, sparkName, sparkDesc, nodes, metrics)

object GlutenSparkPlanGraphCluster {
  def from(cluster: SparkPlanGraphCluster): GlutenSparkPlanGraphCluster = {
    val sparkName = if (cluster.name.contains("WholeStageCodegen")) {
      "WholeStageCodegen"
    } else {
      GlutenParseHelper.mapGlutenToSpark(cluster.name)
    }
    val sparkDesc = if (cluster.desc.contains("WholeStageCodegen")) {
      "WholeStageCodegen"
    } else {
      GlutenParseHelper.mapGlutenToSpark(cluster.desc)
    }
    new GlutenSparkPlanGraphCluster(cluster.id, cluster.name, cluster.desc, sparkName,
      sparkDesc, cluster.nodes, cluster.metrics)
  }
}
