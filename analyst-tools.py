from mcp.server.fastmcp import FastMCP
import pandas as pd
from thefuzz import process
import os
from typing import Optional, Tuple, List, Dict, Union
from mcp.server.fastmcp.prompts import base


mcp = FastMCP("Analyst Tools")


@mcp.tool()
def get_data(metric: str, customer: Optional[str] = None, project: Optional[str] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Get financial metrics at customer or project level. If none is provided, return the overall metric.
    Available metrics: 'gross_margin', 'revenue', 'ebitda'
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "dataset.csv")

    df = pd.read_csv(csv_path, low_memory=False)
    
    # Filter by customer if provided
    if customer:
        customer_names = df['CustomerName'].unique()        
        best_match = process.extractOne(customer, customer_names)        
        if best_match[1] > 80:  # Only use match if confidence > 80%
            customer = best_match[0]
            df = df.loc[df['CustomerName'] == customer]
        else:
            raise ValueError(f"No close match found for customer '{customer}'. Did you mean '{best_match[0]}'?")
    
    # Filter by project if provided
    if project:
        project_names = df['ProjectName'].unique()
        best_match = process.extractOne(project, project_names)
        if best_match[1] > 80:
            project = best_match[0]
            df = df.loc[df['ProjectName'] == project]
        else:
            raise ValueError(f"No close match found for project '{project}'. Did you mean '{best_match[0]}'?")
    
    # Calculate the requested metric
    metric = metric.lower()
    
    # If no customer or project is specified, return the overall metric
    if not customer and not project:
        total_revenue = df['Revenue'].sum()
        total_cogs = df['COGS'].sum()
        
        if metric == 'gross_margin':
            overall_value = total_revenue - total_cogs
            overall_percentage = overall_value / total_revenue if total_revenue > 0 else 0
            result_value = pd.Series([overall_value])
            result_percentage = pd.Series([overall_percentage])
        
        elif metric == 'revenue':
            result_value = pd.Series([total_revenue])
            result_percentage = pd.Series([1.0])  # 100% of revenue
        
        elif metric == 'ebitda':
            total_opex = df['OPEX'].sum()
            overall_value = total_revenue - total_cogs - total_opex
            overall_percentage = overall_value / total_revenue if total_revenue > 0 else 0
            result_value = pd.Series([overall_value])
            result_percentage = pd.Series([overall_percentage])
        
        else:
            raise ValueError(f"Unknown metric: {metric}. Available metrics are 'gross_margin', 'revenue', 'ebitda'.")
            
        return result_value, result_percentage
    
    # Calculate metrics for specific customer/project
    if metric == 'gross_margin':
        result_value = df['Revenue'] - df['COGS']
        result_percentage = result_value / df['Revenue']
    
    elif metric == 'revenue':
        result_value = df['Revenue']
        result_percentage = pd.Series([1.0] * len(df))  # 100% of revenue
    
    elif metric == 'ebitda':
        result_value = df['Revenue'] - df['COGS'] - df['OPEX']
        result_percentage = result_value / df['Revenue']
    
    else:
        raise ValueError(f"Unknown metric: {metric}. Available metrics are 'gross_margin', 'revenue', 'ebitda'.")
    
    return result_value, result_percentage


@mcp.tool()
def compare_performance(entity_type: str, metrics: List[str] = None, top_n: int = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compare performance across different customers or projects for specified metrics.
    
    Parameters:
    - entity_type: Either 'customer' or 'project'
    - metrics: List of metrics to compare, defaults to ['revenue', 'gross_margin', 'ebitda'] if None
    - top_n: If provided, returns only the top N entities by revenue
    
    Returns:
    - Dictionary with entity names as keys, each containing metric values and percentages
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "dataset.csv")

    df = pd.read_csv(csv_path, low_memory=False)
    
    # Validate entity_type
    if entity_type.lower() not in ['customer', 'project']:
        raise ValueError("entity_type must be either 'customer' or 'project'")
    
    # Set default metrics if not provided
    if metrics is None:
        metrics = ['revenue', 'gross_margin', 'ebitda']
    
    # Validate metrics
    valid_metrics = ['revenue', 'gross_margin', 'ebitda']
    for metric in metrics:
        if metric.lower() not in valid_metrics:
            raise ValueError(f"Unknown metric: {metric}. Available metrics are {', '.join(valid_metrics)}")
    
    # Group by the specified entity type
    entity_col = 'CustomerName' if entity_type.lower() == 'customer' else 'ProjectName'
    grouped = df.groupby(entity_col)
    
    # Calculate aggregated metrics for each entity
    results = {}
    
    # Create entity summaries
    entities_summary = []
    
    for entity_name, group in grouped:
        total_revenue = group['Revenue'].sum()
        total_cogs = group['COGS'].sum()
        total_opex = group['OPEX'].sum() if 'OPEX' in df.columns else 0
        
        # Calculate values for each metric
        entity_metrics = {
            'revenue': {
                'value': total_revenue,
                'percentage': 1.0  # 100% of itself
            },
            'gross_margin': {
                'value': total_revenue - total_cogs,
                'percentage': (total_revenue - total_cogs) / total_revenue if total_revenue > 0 else 0
            },
            'ebitda': {
                'value': total_revenue - total_cogs - total_opex,
                'percentage': (total_revenue - total_cogs - total_opex) / total_revenue if total_revenue > 0 else 0
            }
        }
        
        entities_summary.append({
            'name': entity_name,
            'revenue': total_revenue,
            'metrics': {metric: entity_metrics[metric] for metric in metrics}
        })
    
    # Sort by revenue
    entities_summary.sort(key=lambda x: x['revenue'], reverse=True)
    
    # Limit to top_n if specified
    if top_n is not None and isinstance(top_n, int) and top_n > 0:
        entities_summary = entities_summary[:top_n]
    
    # Create final results dictionary
    for entity in entities_summary:
        results[entity['name']] = entity['metrics']
    
    # Also include overall company performance
    total_revenue = df['Revenue'].sum()
    total_cogs = df['COGS'].sum()
    total_opex = df['OPEX'].sum() if 'OPEX' in df.columns else 0
    
    results['OVERALL'] = {
        'revenue': {
            'value': total_revenue,
            'percentage': 1.0
        },
        'gross_margin': {
            'value': total_revenue - total_cogs,
            'percentage': (total_revenue - total_cogs) / total_revenue if total_revenue > 0 else 0
        },
        'ebitda': {
            'value': total_revenue - total_cogs - total_opex,
            'percentage': (total_revenue - total_cogs - total_opex) / total_revenue if total_revenue > 0 else 0
        }
    }
    
    # Filter to only requested metrics
    for entity in results:
        results[entity] = {metric: results[entity][metric] for metric in metrics}
    
    return results


@mcp.resource("entities://all")
def list_all_entities() -> Dict[str, List[str]]:
    """
    Lists all available customers and projects in the dataset.
    
    Returns:
    - Dictionary with information about customers and projects
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "dataset.csv")

    df = pd.read_csv(csv_path, low_memory=False)
    
    # Get unique customer and project names
    customers = sorted(df['CustomerName'].unique().tolist())
    projects = sorted(df['ProjectName'].unique().tolist())
    
    # Create a dictionary with relationships between customers and projects
    customer_projects = {}
    for customer in customers:
        customer_df = df[df['CustomerName'] == customer]
        customer_projects[customer] = sorted(customer_df['ProjectName'].unique().tolist())
    
    project_customers = {}
    for project in projects:
        project_df = df[df['ProjectName'] == project]
        project_customers[project] = sorted(project_df['CustomerName'].unique().tolist())
    
    return {
        'customers': customers,
        'projects': projects,
        'customer_projects': customer_projects,  # Projects associated with each customer
        'project_customers': project_customers   # Customers associated with each project
    }


@mcp.resource("entities://customers")
def list_customers() -> List[str]:
    """
    Lists all available customers in the dataset.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "dataset.csv")

    df = pd.read_csv(csv_path, low_memory=False)
    return sorted(df['CustomerName'].unique().tolist())


@mcp.resource("entities://projects")
def list_projects() -> List[str]:
    """
    Lists all available projects in the dataset.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "dataset.csv")

    df = pd.read_csv(csv_path, low_memory=False)
    return sorted(df['ProjectName'].unique().tolist())


@mcp.resource("entities://customer/{customer_name}/projects")
def list_customer_projects(customer_name: str) -> List[str]:
    """
    Lists all projects associated with a specific customer.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "dataset.csv")

    df = pd.read_csv(csv_path, low_memory=False)
    
    # Find the best match for the customer name
    customer_names = df['CustomerName'].unique()
    best_match = process.extractOne(customer_name, customer_names)
    
    if best_match[1] > 80:  # Only use match if confidence > 80%
        customer = best_match[0]
        customer_df = df[df['CustomerName'] == customer]
        return sorted(customer_df['ProjectName'].unique().tolist())
    else:
        raise ValueError(f"No close match found for customer '{customer_name}'. Did you mean '{best_match[0]}'?")


@mcp.resource("entities://project/{project_name}/customers")
def list_project_customers(project_name: str) -> List[str]:
    """
    Lists all customers associated with a specific project.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "dataset.csv")

    df = pd.read_csv(csv_path, low_memory=False)
    
    # Find the best match for the project name
    project_names = df['ProjectName'].unique()
    best_match = process.extractOne(project_name, project_names)
    
    if best_match[1] > 80:  # Only use match if confidence > 80%
        project = best_match[0]
        project_df = df[df['ProjectName'] == project]
        return sorted(project_df['CustomerName'].unique().tolist())
    else:
        raise ValueError(f"No close match found for project '{project_name}'. Did you mean '{best_match[0]}'?")


@mcp.resource("metrics://available")
def list_available_metrics() -> List[str]:
    """
    Lists all available financial metrics that can be analyzed.
    """
    return ['revenue', 'gross_margin', 'ebitda']


@mcp.prompt()
def financial_metrics_analysis(entity_name: str, metric: str) -> str:
    """Prompt for analyzing a specific financial metric for an entity."""
    return f"""
    As a senior financial analyst, provide a detailed analysis of the {metric} for {entity_name}.
    
    In your analysis:
    1. Assess the current performance level
    2. Compare to industry benchmarks where applicable
    3. Identify key drivers affecting this metric
    4. Recommend strategic actions to optimize this metric
    5. Forecast potential outcomes based on market trends
    
    Use financial terminology appropriately and provide insights that would be valuable for executive decision-making.
    """


@mcp.prompt()
def comparative_financial_analysis(entity_type: str) -> str:
    """Prompt for conducting comparative analysis across entities."""
    return f"""
    As a senior financial analyst, perform a comprehensive comparative analysis of our {entity_type}s.
    
    Your analysis should:
    1. Identify top and bottom performers based on key financial metrics
    2. Analyze performance trends and patterns
    3. Highlight significant outliers and their impact on overall portfolio performance
    4. Assess risk concentration and diversification opportunities
    5. Provide strategic recommendations for portfolio optimization
    
    Support your analysis with relevant financial ratios and metrics. Present your insights in a manner suitable for a quarterly business review with executive leadership.
    """


@mcp.prompt()
def financial_insight_generation(metric: str) -> list[base.Message]:
    """Prompt for generating financial insights about specific metrics with conversation flow."""
    return [
        base.SystemMessage(f"""You are a seasoned financial analyst with expertise in business performance metrics. 
                          Focus on providing actionable insights about {metric} that can drive business decisions.
                          Use appropriate financial terminology and frameworks in your analysis."""),
        base.UserMessage(f"What insights can you provide about our {metric} performance?"),
        base.AssistantMessage(f"""Based on my analysis of your {metric} data, I can offer the following initial observations:
                             
1. The current {metric} performance shows some interesting patterns that warrant deeper investigation.
                             
To provide more tailored insights, I'd need to know if you're interested in a specific customer segment, time period, or business unit comparison."""),
    ]


@mcp.prompt()
def executive_summary_financial() -> str:
    """Prompt for generating executive summary of financial performance."""
    return """
    As the lead financial analyst, prepare an executive summary of the financial performance data.
    
    Your summary should:
    1. Highlight the most critical financial indicators and their implications
    2. Identify key business drivers affecting overall performance
    3. Summarize risks and opportunities revealed by the data
    4. Provide concise, actionable recommendations based on your analysis
    
    Format your response as a concise executive brief that busy C-suite executives can quickly digest while capturing the essential insights they need for strategic decision-making.
    """


@mcp.prompt()
def financial_performance_review() -> str:
    """Prompt for a standard quarterly business review format."""
    return """
    As the senior financial analyst preparing the Quarterly Business Review, analyze the provided financial data and address the following:
    
    PERFORMANCE SUMMARY:
    • Overall revenue performance vs. targets
    • Gross margin and EBITDA trends
    • Key customer/project performance highlights
    
    VARIANCE ANALYSIS:
    • Identify significant variances from forecasts
    • Explain key drivers behind underperforming and overperforming segments
    
    RISK ASSESSMENT:
    • Customer/project concentration risks
    • Margin pressure points
    • Resource allocation efficiency
    
    STRATEGIC RECOMMENDATIONS:
    • Customer/project portfolio optimization
    • Operational efficiency improvements
    • Growth opportunity targeting
    
    Maintain a balanced perspective that acknowledges both positive developments and challenges, with clear prioritization of issues requiring immediate attention.
    """


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"  # Bind to all interfaces, not just localhost
    mcp.run(host=host, port=port)