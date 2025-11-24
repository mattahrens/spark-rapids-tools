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

package com.nvidia.spark.rapids.tool.planparser

import java.nio.file.Paths

import scala.util.matching.Regex

import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.JsonMethods

import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.tool.util.UTF8Source

// Utilities used to handle Auron Ops
object AuronParseHelper extends Logging {
  // Auron operator patterns and mappings
  private val AURON_PATTERN: Regex = "Native[a-zA-Z]*|ConvertToNative|InputAdapter".r
  private val AURON_OPS_MAPPING_DIR = "auronOperatorMappings"
  private val DEFAULT_AURON_OPS_MAPPING_FILE = "auron-default.json"

  // An Auron app is identified using the following properties from spark properties.
  private val SPARK_PROPS_ENABLING_AURON = Map(
    "spark.sql.extensions" -> "org.apache.spark.sql.auron.AuronSparkSessionExtension"
  )

  /**
   * Checks if the properties indicate that the application is an Auron app.
   * This can be checked by looking for Auron-specific properties.
   *
   * @param properties spark properties captured from the eventlog environment details
   * @return true if the properties indicate that it is an Auron app.
   */
  def isAuronApp(properties: collection.Map[String, String]): Boolean = {
    // Check if spark.sql.extensions contains the Auron extension.
    // spark.sql.extensions can contain multiple comma-separated values, so we check if
    // the value contains the Auron extension class name.
    SPARK_PROPS_ENABLING_AURON.exists { case (key, value) =>
      properties.get(key).exists(_.contains(value))
    }
  }

  /**
   * Maps the Auron operator names to Spark operator names using a mapping JSON file.
   */
  private lazy val auronToSparkMapping: Map[String, String] = {
    val mappingFile = Paths.get(AURON_OPS_MAPPING_DIR, DEFAULT_AURON_OPS_MAPPING_FILE).toString
    val jsonString = UTF8Source.fromResource(mappingFile).mkString
    val json = JsonMethods.parse(jsonString)
    // Implicitly define JSON formats for deserialization using DefaultFormats
    implicit val formats: Formats = DefaultFormats
    // Extract and deserialize the JValue object into a Map[String, String]
    // Currently, only the first mapping in the list is used.
    json.extract[Map[String, List[String]]]
      .iterator
      .map { case (k, v) => k -> v.head }
      .toMap
  }

  /**
   * Checks if the node name is an Auron node.
   */
  def isAuronNode(nodeName: String): Boolean = {
    AURON_PATTERN.findFirstIn(nodeName).isDefined
  }

  /**
   * Replaces all occurrences in the input string that match the AURON_PATTERN
   * with corresponding values from the auronToSparkMapping map.
   *
   * @param inputStr the node name, potentially containing an Auron identifier
   * @return the Spark node name
   */
  def mapAuronToSpark(inputStr: String): String = {
    // Handle operators that may have "Exec" suffix (e.g., "NativeProjectExec")
    // Strip "Exec" before mapping, as the reported operator name should not include "Exec"
    val inputWithoutExec = if (inputStr.endsWith("Exec")) {
      inputStr.stripSuffix("Exec")
    } else {
      inputStr
    }

    // Try to match full operator names first (e.g., "NativeParquetScan", "ConvertToNative",
    // "InputAdapter")
    // Sort by length descending to match longer names first (e.g., "NativeParquetScan" before
    // "Native")
    val sortedKeys = auronToSparkMapping.keys.toSeq.sortBy(-_.length)
    val fullMatch = sortedKeys.find(key =>
      inputWithoutExec.startsWith(key) ||
      inputWithoutExec.contains(s"$key ") ||
      inputWithoutExec == key
    )

    val mappedResult = if (fullMatch.isDefined) {
      inputWithoutExec.replace(fullMatch.get, auronToSparkMapping(fullMatch.get))
    } else {
      // Fall back to pattern-based replacement
      AURON_PATTERN.replaceAllIn(inputWithoutExec, m => {
        val matched = m.matched
        auronToSparkMapping.getOrElse(matched, matched)
      })
    }

    // Return the mapped result without "Exec" suffix
    // The "Exec" suffix is added later when needed for checking support (fullExecName)
    mappedResult
  }
}
