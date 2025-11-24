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

package org.apache.spark.sql.rapids.tool.util.plangraph

import scala.collection.mutable

import com.nvidia.spark.rapids.tool.planparser.AuronParseHelper

import org.apache.spark.sql.execution.ui.{SparkPlanGraphCluster, SparkPlanGraphNode, SQLPlanMetric}

/**
 * Extension of SparkPlanGraphNode to handle Auron nodes.
 * Note:
 * - auronName and auronDesc are the name and description of the Auron node
 * - sparkName and sparkDesc are the name and description of the equivalent Spark node
 */
class AuronSparkPlanGraphNode(
    id: Long,
    val auronName: String,
    val auronDesc: String,
    sparkName: String,
    sparkDesc: String,
    metrics: collection.Seq[SQLPlanMetric])
  extends SparkPlanGraphNode(id, sparkName, sparkDesc, metrics)

object AuronSparkPlanGraphNode {
  def from(node: SparkPlanGraphNode): AuronSparkPlanGraphNode = {
    val sparkName = AuronParseHelper.mapAuronToSpark(node.name)
    val sparkDesc = AuronParseHelper.mapAuronToSpark(node.desc)
    new AuronSparkPlanGraphNode(node.id, node.name, node.desc, sparkName, sparkDesc, node.metrics)
  }
}

/**
 * Extension of SparkPlanGraphCluster to handle Auron nodes that are
 * mapped to WholeStageCodegen.
 * Note:
 * - auronName and auronDesc are the name and description of the Auron node
 * - name and desc are the name and description of the equivalent Spark node
 */
class AuronSparkPlanGraphCluster(
    id: Long,
    val auronName: String,
    val auronDesc: String,
    sparkName: String,
    sparkDesc: String,
    nodes: mutable.ArrayBuffer[SparkPlanGraphNode],
    metrics: collection.Seq[SQLPlanMetric])
  extends SparkPlanGraphCluster(id, sparkName, sparkDesc, nodes, metrics)

object AuronSparkPlanGraphCluster {
  def from(cluster: SparkPlanGraphCluster): AuronSparkPlanGraphCluster = {
    val sparkName = AuronParseHelper.mapAuronToSpark(cluster.name)
    val sparkDesc = AuronParseHelper.mapAuronToSpark(cluster.desc)
    new AuronSparkPlanGraphCluster(cluster.id, cluster.name, cluster.desc, sparkName,
      sparkDesc, cluster.nodes, cluster.metrics)
  }
}
