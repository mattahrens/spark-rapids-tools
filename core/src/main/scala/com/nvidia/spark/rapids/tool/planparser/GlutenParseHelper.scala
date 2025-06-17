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

import java.nio.file.Paths

import scala.util.matching.Regex

import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.JsonMethods

import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.tool.util.UTF8Source

// Utilities used to handle Gluten Ops
object GlutenParseHelper extends Logging {
  // A Gluten app is identified using the following properties from spark properties
  private val GLUTEN_SPARK_PROPS = Map(
    "spark.gluten.enabled" -> "true",
    "spark.gluten.sql.columnar.backend.ch" -> ".*",
    "spark.gluten.sql.columnar.backend.velox" -> ".*"
  )

  private val GLUTEN_PATTERN: Regex = "Gluten[a-zA-Z]*".r
  private val GLUTEN_OPS_MAPPING_DIR = "glutenOperatorMappings"
  private val DEFAULT_GLUTEN_OPS_MAPPING_FILE = "gluten-mapping.json"

  /**
   * Checks if the properties indicate that the application is a Gluten app.
   * This can be checked by looking for keywords in one of the keys defined in GLUTEN_SPARK_PROPS
   *
   * @param properties spark properties captured from the eventlog environment details
   * @return true if the properties indicate that it is a Gluten app
   */
  def isGlutenApp(properties: collection.Map[String, String]): Boolean = {
    // First check if Gluten is enabled
    properties.get("spark.gluten.enabled").exists(_ == "true") &&
    // Then check if at least one backend is configured
    GLUTEN_SPARK_PROPS.filterKeys(_ != "spark.gluten.enabled").exists { case (key, value) =>
      properties.get(key).exists(_.matches(value))
    }
  }

  /**
   * Maps the Gluten operator names to Spark operator names using a mapping JSON file.
   */
  private lazy val glutenToSparkMapping: Map[String, String] = {
    val mappingFile = Paths.get(GLUTEN_OPS_MAPPING_DIR, DEFAULT_GLUTEN_OPS_MAPPING_FILE).toString
    val jsonString = UTF8Source.fromResource(mappingFile).mkString
    val json = JsonMethods.parse(jsonString)
    // Implicitly define JSON formats for deserialization using DefaultFormats
    implicit val formats: Formats = DefaultFormats
    // Extract and deserialize the JValue object into a Map[String, String]
    json.extract[Map[String, List[String]]].mapValues(_.head)
  }

  def isGlutenNode(nodeName: String): Boolean = GLUTEN_PATTERN.findFirstIn(nodeName).isDefined

  /**
   * Replaces all occurrences in the input string that match the GLUTEN_PATTERN
   * with corresponding values from the glutenToSparkMapping map.
   *
   * @param inputStr the node name, potentially containing a Gluten identifier
   * @return a String with the Spark node name, or the original string if no match is found
   */
  def mapGlutenToSpark(inputStr: String): String = {
    GLUTEN_PATTERN.replaceAllIn(inputStr, m => glutenToSparkMapping.getOrElse(m.matched, m.matched))
  }
}
