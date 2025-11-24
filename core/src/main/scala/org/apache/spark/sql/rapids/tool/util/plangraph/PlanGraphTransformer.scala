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

import com.nvidia.spark.rapids.tool.planparser.{AuronParseHelper, DatabricksParseHelper, GlutenParseHelper}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.ui.{SparkPlanGraphCluster, SparkPlanGraphNode}


/**
 * Object responsible for transforming instances of SparkPlanGraphNode and SparkPlanGraphCluster
 * to specialized instances (e.g. PhotonSparkPlanGraphNode, AuronSparkPlanGraphNode,
 * GlutenSparkPlanGraphNode) if they are of a specialized type.
 *
 * This can be extended to handle other types of nodes or clusters (e.g. GPU nodes or Velox nodes).
 */
object PlanGraphTransformer extends Logging {

  /**
   * Transforms a SparkPlanGraphNode and returns a specialized node if it is of a specialized
   * type (e.g. Photon, Auron, Gluten).
   */
  def transformPlanNode(node: SparkPlanGraphNode): SparkPlanGraphNode = {
    if (DatabricksParseHelper.isPhotonNode(node.name)) {
      PhotonSparkPlanGraphNode.from(node)
    } else if (AuronParseHelper.isAuronNode(node.name)) {
      AuronSparkPlanGraphNode.from(node)
    } else if (GlutenParseHelper.isGlutenNode(node.name)) {
      logInfo(s"[GLUTEN-DEBUG] transformPlanNode: Transforming Gluten node: " +
        s"'${node.name}' (nodeId: ${node.id})")
      val transformed = GlutenSparkPlanGraphNode.from(node)
      logInfo(s"[GLUTEN-DEBUG] transformPlanNode: Transformed '${node.name}' -> " +
        s"'${transformed.name}'")
      transformed
    } else { // More cases can be added here to handle other types of nodes
      node
    }
  }

  /**
   * Transforms a SparkPlanGraphCluster and returns a specialized cluster if it is of a specialized
   * type (e.g. Photon, Auron, Gluten).
   */
  def transformPlanCluster(cluster: SparkPlanGraphCluster): SparkPlanGraphCluster = {
    if (DatabricksParseHelper.isPhotonNode(cluster.name)) {
      PhotonSparkPlanGraphCluster.from(cluster)
    } else if (AuronParseHelper.isAuronNode(cluster.name)) {
      AuronSparkPlanGraphCluster.from(cluster)
    } else if (GlutenParseHelper.isGlutenNode(cluster.name)) {
      GlutenSparkPlanGraphCluster.from(cluster)
    } else { // More cases can be added here to handle other types of clusters
      cluster
    }
  }
}
